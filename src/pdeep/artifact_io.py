"""Readable manifests and verified, immutable two-phase local artifact commits.

The manifest is the commit record. Payloads are fsynced/read back before either
final file is published. Existing artifacts are never overwritten. A failed
commit removes only its own unpublished files, preserving every previous commit.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json, os, tempfile, shutil
import torch
from .core import canonical_json, digest, require

SCHEMA='pdeep-artifact-v2'
KINDS=('training_checkpoint','inference_export','diagnostic_bundle')


def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def atomic_json(path, value):
    """Atomic replace for small mutable pointers/status, never artifact payloads."""
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.pending-',dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write(canonical_json(value)+'\n');stream.flush();os.fsync(stream.fileno())
        os.replace(tmp,path)
    finally:
        if Path(tmp).exists():Path(tmp).unlink()


def append_event(path,event):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        stream.write(canonical_json(event)+'\n');stream.flush();os.fsync(stream.fileno())


def read_manifest(path, *, kind=None, architecture=None, verify_files=True):
    """JSON inspection needs no deserialization or arbitrary class execution."""
    path=Path(path);record=json.loads(path.read_text())
    require(record.get('schema_version')==SCHEMA,'unsupported artifact schema; use an explicit versioned loader')
    identity=record.get('artifact_id');body={k:v for k,v in record.items() if k!='artifact_id'}
    require(identity==digest(body),'artifact manifest identity/checksum mismatch')
    require(record.get('artifact_type') in KINDS,'unknown artifact type')
    require(record.get('verification',{}).get('status')=='verified','artifact is not verified')
    if kind is not None:require(record['artifact_type']==kind,'artifact type mismatch; inference, diagnostic and resumable artifacts are not interchangeable')
    if architecture is not None:require(record.get('architecture_id')==architecture,'artifact architecture mismatch (B0 and B1 are distinct)')
    require(isinstance(record.get('files'),dict) and len(record['files'])==1,'one self-contained payload is required')
    for name,info in record['files'].items():
        require(Path(name).name==name and name not in {'.','..'},'unsafe artifact payload path')
        payload=path.parent/name
        if verify_files:
            require(payload.is_file() and payload.stat().st_size==info['size_bytes'],'artifact payload size/checksum mismatch')
            require(sha256(payload)==info['sha256'],'artifact payload checksum mismatch')
    return record


def read_payload(path, *, kind=None, architecture=None, required=()):
    record=read_manifest(path,kind=kind,architecture=architecture)
    payload_path=Path(path).parent/next(iter(record['files']))
    payload=torch.load(payload_path,map_location='cpu',weights_only=True)
    require(isinstance(payload,dict) and all(k in payload for k in required),'artifact payload is missing required fields')
    from .artifact_tensors import tree_identity
    require(digest(tree_identity(payload))==record['payload_tree_identity'],'artifact payload tensor/schema identity mismatch')
    return record,payload


def _publish(source,destination):
    """Exclusive hard-link publication on the same local filesystem (no clobber)."""
    os.link(source,destination)


def write_artifact(directory, stem, metadata, payload, *, required=(), idempotent=False):
    from .artifact_tensors import tree_cpu,tree_identity
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    require(metadata.get('artifact_type') in KINDS,'artifact type required')
    metadata=json.loads(canonical_json(metadata))
    payload=tree_cpu(payload);tree_id=digest(tree_identity(payload))
    # Creation time is provenance, not a reason to duplicate identical exports.
    content_id=digest({'metadata':metadata,'payload_tree_identity':tree_id})
    stem=stem or ('artifact-'+content_id)
    require(Path(stem).name==stem and not stem.startswith('.'),'invalid artifact stem')
    manifest=directory/(stem+'.json');target=directory/(stem+'.pt')
    lock=directory/(stem+'.lock');fd=None;temps=[];published=[]
    try:
        fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
        require(not target.exists() and not manifest.exists() or idempotent,'checkpoint exists; select a new update/run or explicitly load it')
        if target.exists() or manifest.exists():
            old=read_manifest(manifest)
            require(old['content_identity']==content_id,'existing artifact has different scientific content')
            return manifest
        pfd,tmp=tempfile.mkstemp(prefix='.pending-',dir=directory);temps.append(Path(tmp))
        with os.fdopen(pfd,'wb') as stream:
            torch.save(payload,stream);stream.flush();os.fsync(stream.fileno())
        checksum=sha256(tmp);size=Path(tmp).stat().st_size
        reopened=torch.load(tmp,map_location='cpu',weights_only=True)
        require(isinstance(reopened,dict) and all(k in reopened for k in required),'written payload missing required fields')
        require(digest(tree_identity(reopened))==tree_id,'written payload failed read-back identity verification')
        body={**metadata,'schema_version':SCHEMA,'content_identity':content_id,'payload_tree_identity':tree_id,
              'created_UTC':datetime.now(timezone.utc).isoformat(),
              'files':{target.name:{'sha256':checksum,'size_bytes':size}},
              'verification':{'status':'verified','local_payload_readback':True,'checksum_verified':True,
                              'commit_protocol':'fsync temporary payload/readback/checksum; exclusive payload then manifest commit'}}
        record={**body,'artifact_id':digest(body)}
        mfd,mtmp=tempfile.mkstemp(prefix='.pending-',dir=directory);temps.append(Path(mtmp))
        with os.fdopen(mfd,'w') as stream:
            stream.write(json.dumps(record,indent=2,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno())
        # Verify JSON independently before publishing the commit record.
        require(json.loads(Path(mtmp).read_text())==record,'manifest readback mismatch')
        _publish(tmp,target);published.append(target)
        _publish(mtmp,manifest);published.append(manifest)
        read_manifest(manifest)
        return manifest
    except Exception:
        # Previous stems/pointers were never touched. A new partial transaction
        # cannot block recovery by leaving an uncommitted final payload behind.
        for path in reversed(published):
            if path.exists():path.unlink()
        raise
    finally:
        for path in temps:
            if path.exists():path.unlink()
        if fd is not None:
            os.close(fd)
            if lock.exists():lock.unlink()

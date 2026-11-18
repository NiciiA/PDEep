"""Ordinary filesystem transfer with explicit inventories and verified commits.

Mounted Drive is not assumed to provide local POSIX atomicity. A partial marker
precedes copying; complete is published only after verified destination contents.
No directory discovery, remote API, database, or reference solve is involved.
"""
from pathlib import Path
import json,os,shutil,tempfile,time,tarfile
from ..core import require,digest
from ..artifact_io import sha256,atomic_json,read_manifest,read_payload
from .portable import read_index,verify_folder,relative,commit,resolve_manifest,LAYOUT,add_file,add_artifact

def copy_file(source,target,expected):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.copying-',dir=target.parent);os.close(fd)
    try:
        with open(source,'rb') as src,open(tmp,'wb') as dst:
            shutil.copyfileobj(src,dst,length=4*1024*1024);dst.flush();os.fsync(dst.fileno())
        require(Path(tmp).stat().st_size==expected['size_bytes'] and sha256(tmp)==expected['sha256'],'transfer temporary file checksum failed')
        os.replace(tmp,target)
        require(target.stat().st_size==expected['size_bytes'] and sha256(target)==expected['sha256'],'transfer destination checksum failed')
    finally:
        if Path(tmp).exists():Path(tmp).unlink()

def sync_folder(source,destination,*,progress=None,max_files=None,verified_cache=None):
    start=time.perf_counter();source,index=read_index(source);destination=Path(destination).expanduser()
    require(source.parent.resolve()!=destination.resolve(),'source and destination dataset folders are identical')
    destination.mkdir(parents=True,exist_ok=True);target=destination/'manifest.json';cache={} if verified_cache is None else verified_cache
    owner=('destination',);where=str(destination.resolve())
    require(owner not in cache or cache[owner]==where,'verified transfer cache belongs to another destination')
    cache[owner]=where
    if target.exists():
        _,old=read_index(target)
        require((old['dataset_identity'],old['certificate_id'],old['inventory_identity'])==(index['dataset_identity'],index['certificate_id'],index['inventory_identity']),'incompatible destination dataset; choose another folder')
        removed={k:v for k,v in old['files'].items() if k not in index['files']}
        consumed=lambda name:any(name.startswith('.working/'+cid+'/') for cid in old.get('partial_cases',{}) if cid in index['records'])
        require(all(index['files'].get(k)==v or k in removed and consumed(k) for k,v in old['files'].items()),'destination snapshot contains conflicting/newer files; do not overwrite')
    else:removed={}
    pending={**index,'status':'partial','transfer_source_manifest':index['manifest_identity']};commit(target,pending)
    copied=skipped=bytes_copied=0
    for rel,info in index['files'].items():
        # Generation owns immutable artifacts. Within one live sync session,
        # previously verified files need no repeated remote stat/hash requests.
        # A fresh process verifies again; completion verifies the entire folder.
        if rel in cache and cache[rel][1]==info['sha256']:skipped+=1;continue
        src=relative(source.parent,rel);dst=relative(destination,rel)
        require(src.is_file() and src.stat().st_size==info['size_bytes'],'source file missing: '+rel)
        stamp=(dst.stat().st_size,dst.stat().st_mtime_ns) if dst.is_file() else None
        if stamp is not None and (cache.get(rel)==(stamp,info['sha256']) or stamp[0]==info['size_bytes'] and sha256(dst)==info['sha256']):
            cache[rel]=(stamp,info['sha256']);skipped+=1;continue
        if max_files is not None and copied>=max_files:return {'status':'partial','transfer_status':'interrupted','dataset_status':'partial','manifest':str(target),'copied_files':copied,'skipped_verified_files':skipped,'seconds':time.perf_counter()-start}
        copy_file(src,dst,info);copied+=1;bytes_copied+=info['size_bytes'];cache[rel]=((dst.stat().st_size,dst.stat().st_mtime_ns),info['sha256'])
        if progress:progress('Dataset transfer',copied_files=copied,skipped_verified_files=skipped,total_files=len(index['files']),bytes_copied=bytes_copied)
    # Full destination verification once at completion. Incremental preparation
    # sync uses partial snapshots and the in-process verified cache above.
    if index['status']=='complete':
        try:verify_folder(target)
        except Exception:cache.clear();raise
    commit(target,dict(index))
    require(read_index(target)[1]['manifest_identity']==index['manifest_identity'],'destination manifest readback failed')
    # A completed case replaces its temporary label shards. Delete ONLY the
    # explicitly recorded, checksum-matching scratch files after final commit.
    for rel,info in removed.items():
        p=relative(destination,rel)
        if p.is_file() and sha256(p)==info['sha256']:p.unlink()
    return {'status':index['status'],'transfer_status':'verified','dataset_status':index['status'],'manifest':str(target),'copied_files':copied,'skipped_verified_files':skipped,'bytes_copied':bytes_copied,'seconds':time.perf_counter()-start,'verification':'per-file checksums; complete manifest published last; mounted-Drive atomicity not assumed'}

def archive_dataset(source,archive):
    start=time.perf_counter();source,index=verify_folder(source,require_complete=True);archive=Path(archive).expanduser()
    require(not archive.resolve().is_relative_to(source.parent.resolve()),'put transfer archive outside dataset folder')
    require(not archive.exists(),'archive exists; choose a new output path');archive.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.archiving-',dir=archive.parent);os.close(fd)
    try:
        with tarfile.open(tmp,'w:gz',compresslevel=1,dereference=True) as tar:
            tar.add(source,arcname='manifest.json',recursive=False)
            for rel in sorted(index['files']):tar.add(relative(source.parent,rel),arcname=rel,recursive=False)
        os.replace(tmp,archive)
        meta={'format':'G1-verified-tar-gz-v1','dataset_identity':index['dataset_identity'],'manifest_identity':index['manifest_identity'],'sha256':sha256(archive),'size_bytes':archive.stat().st_size,'uncompressed_file_bytes':sum(v['size_bytes'] for v in index['files'].values()),'seconds':time.perf_counter()-start}
        atomic_json(str(archive)+'.json',meta);return meta
    finally:
        if Path(tmp).exists():Path(tmp).unlink()

def extract_archive(archive,destination):
    start=time.perf_counter();archive=Path(archive).expanduser();side=Path(str(archive)+'.json')
    require(side.is_file(),'archive checksum sidecar missing');meta=json.loads(side.read_text())
    require(meta.get('format')=='G1-verified-tar-gz-v1' and archive.stat().st_size==meta['size_bytes'] and sha256(archive)==meta['sha256'],'archive checksum mismatch')
    destination=Path(destination).expanduser();destination.mkdir(parents=True,exist_ok=True);target=destination/'manifest.json'
    with tarfile.open(archive,'r:gz') as tar:
        members=tar.getmembers();names=[m.name for m in members];require(len(names)==len(set(names)) and all(m.isfile() for m in members),'unsafe archive entries')
        require('manifest.json' in names,'archive manifest missing');data=json.load(tar.extractfile('manifest.json'))
        if target.exists():
            _,old=read_index(target);require(old['dataset_identity']==data['dataset_identity'] and all(data['files'].get(k)==v for k,v in old['files'].items()),'archive destination conflict')
        require(data['manifest_identity']==meta['manifest_identity'] and data['manifest_identity']==digest({k:v for k,v in data.items() if k!='manifest_identity'}),'archive manifest identity mismatch')
        require(data['dataset_identity']==meta['dataset_identity'],'archive dataset identity mismatch')
        for rel in data['files']:relative(destination,rel)
        pending={**data,'status':'partial'};commit(target,pending);read_index(target)
        require(set(names)==set(data['files'])|{'manifest.json'},'archive inventory differs')
        copied=skipped=0
        # Physical tar order matters for gzip: arbitrary manifest order can seek
        # backwards and repeatedly decompress multi-GB prefixes. Member objects
        # also avoid an O(number_of_files) name lookup for every extracted file.
        for member in members:
            rel=member.name
            if rel=='manifest.json':continue
            expected=data['files'][rel]
            dst=relative(destination,rel)
            if dst.is_file() and dst.stat().st_size==expected['size_bytes'] and sha256(dst)==expected['sha256']:skipped+=1;continue
            dst.parent.mkdir(parents=True,exist_ok=True);fd,tmp=tempfile.mkstemp(prefix='.extracting-',dir=dst.parent)
            try:
                with os.fdopen(fd,'wb') as stream,tar.extractfile(member) as content:shutil.copyfileobj(content,stream);stream.flush();os.fsync(stream.fileno())
                require(Path(tmp).stat().st_size==expected['size_bytes'] and sha256(tmp)==expected['sha256'],'archive member checksum mismatch: '+rel)
                os.replace(tmp,dst);copied+=1
            finally:
                if Path(tmp).exists():Path(tmp).unlink()
    verify_folder(target);commit(target,dict(data));verify_folder(target,require_complete=True)
    return {'status':'complete','manifest':str(target),'copied_files':copied,'skipped_verified_files':skipped,'seconds':time.perf_counter()-start}

def export_legacy(source,destination,certificate):
    """Copy only explicit legacy records/assets; original manifests never rewritten."""
    from ..unified.config import DatasetConfig
    from ..unified.streams import STREAM_SCHEMA
    from .portable import compatible_certificates
    source=resolve_manifest(source);old=json.loads(source.read_text());config=DatasetConfig(**old['config']);destination=Path(destination).expanduser()
    require(old['dataset_identity']==config.identity and old['certificate_id']==certificate['certification_id'],'legacy configuration/producer certificate mismatch')
    target=destination/'manifest.json'
    previous=None
    if target.exists():
        _,previous=read_index(target);require(previous['dataset_identity']==config.identity and previous['certificate_id']==certificate['certification_id'],'incompatible legacy-export destination')
    destination.mkdir(parents=True,exist_ok=True)
    certfile=destination/'quality_reports/producer-certificate.json';atomic_json(certfile,certificate)
    index=previous or {k:v for k,v in old.items() if k!='persistence'}
    if previous is None:index.update(portable_layout=LAYOUT,files={},records={},assets={},producer_certificate=certfile.relative_to(destination).as_posix(),persistence={},status='partial')
    add_file(index,destination,certfile)
    paths=set(old['records'].values())|set(old['assets'].values())
    for rel in sorted(paths):
        local=source.parent.parent.parent/rel
        if not local.is_file():
            saved=next((v for k,v in old.get('persistence',{}).items() if k.endswith('/'+rel)),None)
            require(saved and saved.get('Drive_verified'),'missing explicit legacy artifact pointer: '+rel);local=Path(saved['Drive_manifest'])
        record=read_manifest(local);dst=relative(destination,rel)
        for name,expected in record['files'].items():
            copied=dst.parent/name;old_info=index['files'].get(copied.relative_to(destination).as_posix())
            require(old_info is None or old_info==expected,'conflicting legacy-export artifact')
            if not copied.is_file() or copied.stat().st_size!=expected['size_bytes'] or sha256(copied)!=expected['sha256']:copy_file(local.parent/name,copied,expected)
        expected={'sha256':sha256(local),'size_bytes':local.stat().st_size};old_info=index['files'].get(rel)
        require(old_info is None or old_info==expected,'conflicting legacy-export manifest')
        if not dst.is_file() or sha256(dst)!=expected['sha256']:copy_file(local,dst,expected)
        add_artifact(index,destination,dst)
        for cid,value in old['records'].items():
            if value==rel:index['records'][cid]=rel
        for key,value in old['assets'].items():
            if value==rel:index['assets'][key]=rel
        commit(target,index)
    index['status']='complete' if len(index['records'])==sum(config.totals.values()) and not index['quarantine'] and all(index['qualification'].get('/'.join(t),{}).get('passed') for t in config.tasks) else 'partial'
    commit(target,index);verify_folder(target);return target

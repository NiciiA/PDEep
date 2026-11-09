"""Optional verified mounted-Drive copies. Local commits remain authoritative.

No credentials, mounting, network API or deletion is attempted. A filesystem_test
backend exercises the same transaction but is never labeled actual Drive testing.
"""
from dataclasses import dataclass,asdict
from pathlib import Path
import os,shutil,tempfile,time,importlib.util
from .core import require,digest
from .artifact_io import read_manifest,sha256

@dataclass(frozen=True)
class PersistencePolicy:
    drive_enabled: bool=False
    drive_mount: str | None=None
    drive_explicitly_mounted: bool=False
    backend: str='colab_drive'
    retention: str='keep_all'
    keep_last: int=5
    drive_root: str | None=None  # Optional configured folder inside drive_mount.

    def __post_init__(self):
        require(self.backend in {'colab_drive','filesystem_test'},'unsupported persistence backend')
        require(self.retention in {'keep_all','suggest_keep_last'} and self.keep_last>0,'invalid non-destructive retention policy')
        if self.drive_enabled:require(self.drive_mount is not None and self.drive_explicitly_mounted,'Drive must be explicitly enabled and already mounted')
        if self.drive_root is not None:
            require(self.drive_mount is not None and Path(self.drive_root).resolve().is_relative_to(Path(self.drive_mount).resolve()),
                    'configured Drive root must be inside the mounted persistence root')
    @property
    def identity(self):return digest(asdict(self))


def sync_verified(manifest,policy):
    started=time.monotonic();manifest=Path(manifest);record=read_manifest(manifest)
    status={'backend':policy.backend,'local_manifest':str(manifest),'local_verified':True,'Drive_enabled':policy.drive_enabled,
            'Drive_status':'disabled','Drive_manifest':None,'Drive_verified':False,'seconds':0.,'error':None}
    if not policy.drive_enabled:return status
    temporary=None
    try:
        mount=Path(policy.drive_mount)
        require(mount.is_dir(),'explicitly mounted persistence root is unavailable')
        if policy.backend=='colab_drive':
            try:colab=importlib.util.find_spec('google.colab') is not None
            except ModuleNotFoundError:colab=False
            require(colab and os.path.ismount(mount),'real Drive backend requires Colab and an actual mounted Drive root; no mount performed')
        base=mount/'PDEep_artifacts' if policy.drive_root is None else Path(policy.drive_root)/'checkpoints'
        destination=base/record['run_id']/record['artifact_id']
        destination.parent.mkdir(parents=True,exist_ok=True)
        if destination.exists():
            copied=read_manifest(destination/manifest.name);require(copied['artifact_id']==record['artifact_id'],'existing remote artifact identity mismatch')
        else:
            temporary=Path(tempfile.mkdtemp(prefix='.copying-',dir=destination.parent))
            for name,info in record['files'].items():
                copied=temporary/name;shutil.copyfile(manifest.parent/name,copied)
                require(copied.stat().st_size==info['size_bytes'] and sha256(copied)==info['sha256'],'Drive copy size/checksum verification failed')
            shutil.copyfile(manifest,temporary/manifest.name);read_manifest(temporary/manifest.name)
            os.rename(temporary,destination);temporary=None
        status.update(Drive_status='completed',Drive_manifest=str(destination/manifest.name),Drive_verified=True)
    except Exception as error:status.update(Drive_status='failed',error=f'{type(error).__name__}: {error}')
    finally:
        if temporary is not None:
            try:
                if temporary.exists():shutil.rmtree(temporary)
            except OSError as cleanup_error:
                # A disconnected mount can fail BOTH copy and cleanup. Keep the
                # verified local artifact authoritative and report the orphan;
                # never let a remote cleanup error interrupt training.
                status['cleanup_error']=str(cleanup_error)
                status['unfinalized_temporary_destination']=str(temporary)
        status['seconds']=time.monotonic()-started
    return status


def retention_candidates(manifests,policy):
    """A preview only. Never removes a checkpoint or experiment."""
    paths=sorted(map(Path,manifests))
    return [] if policy.retention=='keep_all' else paths[:-policy.keep_last]

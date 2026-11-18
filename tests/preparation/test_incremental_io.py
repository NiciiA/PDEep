"""Filesystem-session cache control, no numerical dataset or solver."""
import json
from pathlib import Path
import pytest
from pdeep.artifact_io import sha256
from pdeep.preparation.portable import commit
from pdeep.preparation.transfer import sync_folder

def test_live_incremental_sync_avoids_rechecking_immutable_remote_files(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();content=source/'tiny.dat';content.write_bytes(b'Verified immutable control')
    d={'dataset_identity':'IO-unit-control','certificate_id':'IO-unit-control','inventory_identity':'IO-unit-control','status':'partial','records':{},'files':{'tiny.dat':{'sha256':sha256(content),'size_bytes':content.stat().st_size}}};commit(source/'manifest.json',d)
    # Test only transport caching. Scientific manifest validation has independent
    # real-P4 round-trip tests and is intentionally not part of this IO control.
    def read(path):
        path=Path(path);path=path/'manifest.json' if path.is_dir() else path;return path,json.loads(path.read_text())
    monkeypatch.setattr('pdeep.preparation.transfer.read_index',read)
    cache={};first=sync_folder(source,tmp_path/'destination',verified_cache=cache);assert first['copied_files']==1
    stat=Path.stat
    def audited(path,*a,**k):
        if path.name=='tiny.dat':pytest.fail('cached immutable file was unnecessarily restatted')
        return stat(path,*a,**k)
    monkeypatch.setattr(Path,'stat',audited)
    second=sync_folder(source,tmp_path/'destination',verified_cache=cache)
    assert second['copied_files']==0 and second['skipped_verified_files']==1
    with pytest.raises(ValueError,match='another destination'):sync_folder(source,tmp_path/'other',verified_cache=cache)

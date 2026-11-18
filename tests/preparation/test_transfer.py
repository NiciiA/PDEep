from pathlib import Path
from dataclasses import replace
import json,shutil,tarfile,io
import numpy as np
import pytest
from pdeep.core import digest
from pdeep.artifact_io import atomic_json
from pdeep.preparation.portable import Dataset,verify_folder,read_index,commit,compatible_certificates
from pdeep.preparation.transfer import sync_folder,archive_dataset,extract_archive,export_legacy
from pdeep.preparation.engine import prepare_dataset
from pdeep.preparation.workflow import prepare
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.smooth.reference import Reference
from pdeep.unified.data import Provider

def forbidden(*a,**k):pytest.fail('completed data attempted reference regeneration or neural optimizer construction')

def test_portable_root_copy_load_and_no_regeneration(prepared,certificate,tmp_path,monkeypatch):
    import torch
    destination=tmp_path/'another-machine/G1_unified/datasets/research_v1'
    result=sync_folder(prepared.path,destination);assert result['status']=='complete'
    monkeypatch.setattr(FixedReference,'__init__',forbidden);monkeypatch.setattr(Reference,'__init__',forbidden);monkeypatch.setattr(torch.optim.AdamW,'__init__',forbidden)
    data=Dataset(destination,certificate);data.require_ready('train');data.require_ready('validation');data.require_ready('test')
    c=data.cases('train')[0];a,_=Provider(data).batch((c,),count=16,observation_count=128);b,_=Provider(prepared).batch((c,),count=16,observation_count=128)
    np.testing.assert_array_equal(a.queries.value,b.queries.value)
    assert data.config.identity==prepared.config.identity and data.index['records']==prepared.index['records']
    repeat=sync_folder(prepared.path,destination);assert repeat['copied_files']==0 and repeat['skipped_verified_files']==len(prepared.index['files'])
    loaded=prepare(prepared.config,tmp_path/'colab-local',certificate,action='restore',manifest=destination)
    assert loaded.root.resolve().is_relative_to((tmp_path/'colab-local').resolve())
    again=prepare_dataset(data.config,destination,certificate,progress=lambda *a,**k:None);assert again.index['status']=='complete'

def test_interrupted_corrupted_missing_and_identity(prepared,certificate,tmp_path):
    dest=tmp_path/'interrupted';first=sync_folder(prepared.path,dest,max_files=2);assert first['status']=='partial'
    with pytest.raises(ValueError,match='missing/corrupt'):Dataset(dest,certificate)
    resumed=sync_folder(prepared.path,dest);assert resumed['skipped_verified_files']==2;verify_folder(dest,require_complete=True)
    rel=next(k for k in prepared.index['files'] if k.endswith('.pt'));file=dest/rel;file.write_bytes(b'corrupted')
    with pytest.raises(ValueError,match='corrupt'):verify_folder(dest)
    repaired=sync_folder(prepared.path,dest);assert repaired['copied_files']==1
    file.unlink()
    with pytest.raises(ValueError,match='missing'):verify_folder(dest)
    sync_folder(prepared.path,dest)
    _,index=read_index(dest);index['certificate_id']='wrong';commit(dest/'manifest.json',index)
    with pytest.raises(ValueError,match='incompatible destination'):sync_folder(prepared.path,dest)

def test_manifest_integrity_and_path_safety(prepared,tmp_path):
    dest=tmp_path/'unsafe';sync_folder(prepared.path,dest);path=dest/'manifest.json';d=json.loads(path.read_text());d['records']['fake']='../../outside.json';commit(path,d)
    with pytest.raises(ValueError):read_index(path)

def test_archive_roundtrip_and_corruption(prepared,tmp_path):
    archive=tmp_path/'dataset.tar.gz';meta=archive_dataset(prepared.path,archive);dest=tmp_path/'extracted';result=extract_archive(archive,dest)
    assert result['status']=='complete';verify_folder(dest,require_complete=True)
    assert extract_archive(archive,dest)['copied_files']==0
    archive.write_bytes(archive.read_bytes()+b'x')
    with pytest.raises(ValueError,match='checksum'):extract_archive(archive,tmp_path/'bad')

def test_preview_no_generation_or_drive(certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(FixedReference,'__init__',forbidden)
    assert prepare(None,tmp_path,certificate,action='preview') is None

def test_actual_issued_producer_certificate_compatibility(certificate,prepared):
    from pdeep.learning_diagnostics.certificates import resolve_certification
    issued=resolve_certification().record
    if issued is None:pytest.skip('full issued certificate absent; no external qualification claim')
    compatible_certificates(issued,certificate,prepared.config)
    wrong=json.loads(json.dumps(issued));wrong['reference_policy']['reference_dt']=.02;wrong['certification_id']=digest({k:v for k,v in wrong.items() if k!='certification_id'})
    with pytest.raises(ValueError,match='scientific mismatch'):compatible_certificates(wrong,certificate,prepared.config)

def test_legacy_export_interruption_and_bytes_preserved(prepared,certificate,tmp_path,monkeypatch):
    import pdeep.preparation.transfer as transfer
    legacy=tmp_path/'legacy-global';index=legacy/'datasets'/prepared.config.identity/'index.json'
    old={k:v for k,v in prepared.index.items() if k not in ('files','portable_layout','producer_certificate','manifest_identity','status')}
    for rel in set(old['records'].values())|set(old['assets'].values()):
        source=prepared.root/rel;r=__import__('pdeep.artifact_io',fromlist=['read_manifest']).read_manifest(source)
        dst=legacy/rel;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(source,dst)
        for name in r['files']:shutil.copyfile(source.parent/name,dst.parent/name)
    atomic_json(index,old);destination=tmp_path/'portable-export';original=transfer.copy_file;calls=[]
    def interrupted(*args,**kwargs):
        calls.append(1)
        if len(calls)==3:raise OSError('simulated disconnect')
        return original(*args,**kwargs)
    with monkeypatch.context() as m:
        m.setattr(transfer,'copy_file',interrupted)
        with pytest.raises(OSError,match='disconnect'):export_legacy(index,destination,certificate)
    assert read_index(destination)[1]['status']=='partial'
    path=export_legacy(index,destination,certificate);verify_folder(path,require_complete=True)
    copied=Dataset(path,certificate);assert copied.index['records']==prepared.index['records']
    from pdeep.artifact_io import sha256
    for rel in old['records'].values():assert sha256(copied.root/rel)==sha256(prepared.root/rel)

def test_cli_verify_returns_success(prepared,capsys):
    from pdeep.preparation.__main__ import main
    assert main(['verify',str(prepared.path),'--complete'])==0
    assert json.loads(capsys.readouterr().out)['status']=='complete'

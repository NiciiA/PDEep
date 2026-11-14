from dataclasses import asdict
from pathlib import Path
import ast,json
import numpy as np
import pytest
import torch
from pdeep.core import digest,ValidationError
from pdeep.artifact_io import write_artifact,atomic_json,read_manifest
from pdeep.artifact_drive import PersistencePolicy
from pdeep.unified.config import DatasetConfig
from pdeep.unified.dataset import inventory,prepare_dataset
from pdeep.unified.persistence import DatasetPersistence,restore_dataset,copy_verified
from .conftest import EVIDENCE


def test_known_dataset_restore_filesystem_backend_no_search(tmp_path,monkeypatch):
    local=tmp_path/'local';drive=tmp_path/'drive';drive.mkdir();d=DatasetConfig(preset='smoke');persistence=DatasetPersistence(local,d.identity,PersistencePolicy(True,str(drive),True,backend='filesystem_test',drive_root=str(drive)))
    row=inventory(d)[0]
    a=write_artifact(local/'responses','asset',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'responses','pde':'poisson','family':'square'},{'tensor':torch.ones(2)})
    c=write_artifact(local/'cases','case',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'cases'},{'case_id':row['case_id']})
    index={'config':asdict(d),'dataset_identity':d.identity,'cases':[row],'records':{row['case_id']:'cases/case.json'},'assets':{'asset':'responses/asset.json'},'persistence':{str(a):persistence(a),str(c):persistence(c)}}
    path=local/'datasets'/d.identity/'index.json';atomic_json(path,index);status=persistence(path,index_only=True)
    assert status['Drive_verified'] and index['persistence'][str(a)]['backend']=='filesystem_test'
    original_glob=Path.glob;original_rglob=Path.rglob
    def guard(fn):
        def call(p,*a,**kw):
            if p.is_relative_to(drive):pytest.fail('Drive search forbidden')
            return fn(p,*a,**kw)
        return call
    monkeypatch.setattr(Path,'glob',guard(original_glob));monkeypatch.setattr(Path,'rglob',guard(original_rglob))
    restored=restore_dataset(status['Drive_index'],tmp_path/'restored',case_ids=[row['case_id']]);assert restored.exists()
    assert read_manifest(tmp_path/'restored/responses/asset.json')['artifact_id']==read_manifest(a)['artifact_id']
    (drive/'unavailable').mkdir();broken=DatasetPersistence(local,d.identity,PersistencePolicy(True,str(drive/'unavailable'),True,backend='filesystem_test'))
    (drive/'unavailable').rmdir()
    with pytest.warns(RuntimeWarning,match='DRIVE SYNC FAILED'):failure=broken(a)
    assert failure['Drive_status']=='failed' and read_manifest(a)['verification']['status']=='verified'
    EVIDENCE['Drive_testing']='verified local filesystem_test copies/restore/failure; actual Colab/Drive NOT tested'


def test_failed_new_reference_quarantined_no_training(certificate,tmp_path,monkeypatch):
    import pdeep.unified.dataset as mod
    monkeypatch.setattr('pdeep.learning_diagnostics.reference_cache.FixedReference',lambda *a,**k:object())
    monkeypatch.setattr(mod,'quality_check_full',lambda *a,**kw:{'passed':False,'checks':{'finite':False}})
    d=prepare_dataset(DatasetConfig(preset='smoke'),tmp_path,certificate,max_cases=1,progress=lambda *a,**kw:None)
    assert len(d.index['quarantine'])==1 and not d.index['records']
    with pytest.raises(ValidationError,match='unprepared or quarantined'):d.require_case(d.cases('train')[0])
    with pytest.raises(ValidationError):d.require_ready()


def test_new_notebook_safe_compact_independent_defaults():
    import nbformat
    root=Path(__file__).resolve().parents[2];n=nbformat.read(root/'notebooks/04_g1_unified.ipynb',4);nbformat.validate(n)
    assert len(n.cells)<=20
    code='\n'.join(c.source for c in n.cells if c.cell_type=='code');ast.parse(code)
    for literal in ('G1_RUN_MODE = "preview"','G1_DATA_ACTION = "preview"','G1_RUN_PREFLIGHT = False','G1_EVALUATE = False','G1_CHECKPOINT_LOOKUP = "none"','G1_TARGET_SUCCESSFUL_UPDATES = 30000','G1_STOP_AFTER_UPDATES = 3000'):
        assert literal in code
    assert '02_train.ipynb' not in code and '03_evaluate.ipynb' not in code
    assert code.count('G1_RESULT = train_models(')==1
    assert 'ensure_compatible_certificate(' not in code and 'weights_only=False' not in code

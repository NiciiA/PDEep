from pathlib import Path
import json,time
import pytest,torch
from pdeep.learning_diagnostics.certificates import resolve_certification
from tests.unified.conftest import ArithmeticDataset
from pdeep.unified_modes.config import TrainConfig

EVIDENCE={}
@pytest.fixture(scope='session',autouse=True)
def resources():
    old=torch.get_num_threads();torch.set_num_threads(1);start=time.perf_counter();yield;torch.set_num_threads(old)
    p=Path(__file__).resolve().parents[2]/'reports/g1_modes/checks.json';p.parent.mkdir(parents=True,exist_ok=True)
    old=json.loads(p.read_text()) if p.exists() else {};m=old.get('measurements',{});m.update(EVIDENCE)
    p.write_text(json.dumps({'scope':'bounded arithmetic/numerical controls; no research training or recertification','seconds':time.perf_counter()-start,'measurements':m},indent=2)+'\n')
@pytest.fixture(scope='session')
def certificate():
    r=resolve_certification(requested_capabilities={'certification_id':None})
    if not r.verified:pytest.skip('native certificate unavailable; no recertification')
    return r.require()
@pytest.fixture
def dataset(certificate):return ArithmeticDataset(certificate)
def config(d,model='C_GEO/FIELD_ONLY',**kw):
    values=dict(architecture=model.split('/')[0],model_key=model,preset='smoke',capacity='verification',device='cpu',dataset_identity=d.config.identity,experiment_id='bounded-modes-fixture',updates=3,main_target=2,warmup_updates=0,scheduler='constant',query_count=8,query_chunk=5,validation_query_count=8,validation_every=500,checkpoint_every=500)
    values.update(kw);return TrainConfig(**values)

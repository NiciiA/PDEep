"""Bounded tests: small numerical controls and untrained model arithmetic fixtures."""
from types import SimpleNamespace
from pathlib import Path
from dataclasses import replace
import json,time
import numpy as np
import pytest
import torch
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.smooth.certification import normalizers
from pdeep.smooth.data import Case
from pdeep.unified.config import DatasetConfig,TrainConfig
from pdeep.unified.data import Provider

EVIDENCE={}

@pytest.fixture(scope='session',autouse=True)
def bounded_cpu_threads():
    previous=torch.get_num_threads();torch.set_num_threads(1);start=time.perf_counter()
    yield
    torch.set_num_threads(previous)
    root=Path(__file__).resolve().parents[2]/'reports/g1_unified';root.mkdir(exist_ok=True,parents=True)
    (root/'bounded_checks.json').write_text(json.dumps({'scope':'bounded CPU controls; no research training or numerical recertification',
        'seconds':time.perf_counter()-start,'measurements':EVIDENCE},indent=2)+'\n')

@pytest.fixture(scope='session')
def certificate():
    r=resolve_certification(requested_capabilities={'certification_id':None})
    if not r.verified:pytest.skip('existing native certificate unavailable; never auto-qualify')
    return r.require()

class ArithmeticCache:
    """ZERO targets solely for encoder/loss/engine arithmetic, NOT a reference claim."""
    def evaluate(self,c,x,t,q,**kwargs):return np.zeros(len(x),np.float64)

class ArithmeticDataset:
    def __init__(self,certificate,config=None):
        self.config=config or DatasetConfig(preset='smoke');self.certificate=certificate;self.norms=normalizers(certificate)
        self.cache=ArithmeticCache();self.path=Path('/arithmetic-test-fixture-not-a-certified-dataset.json')
    def require_ready(self,*a):pass
    def require_case(self,c):pass
    def cases(self,split,**kw):
        from pdeep.unified.dataset import case_from_key
        return tuple(case_from_key(k) for k in self.config.keys(split))

@pytest.fixture
def arithmetic_dataset(certificate):return ArithmeticDataset(certificate)

@pytest.fixture
def batch(arithmetic_dataset):
    cases=(Case('square','poisson','train',0),Case('disk','heat','train',0),Case('rectangle','wave','train',1),Case('hole','wave','train',2))
    return Provider(arithmetic_dataset).batch(cases,count=8,seed=42)[0]


def config(dataset,model='A_GEO/KIN_001',**kw):
    return TrainConfig(architecture=model.split('/')[0],model_key=model,preset='smoke',capacity='verification',device='cpu',
        dataset_identity=dataset.config.identity,experiment_id='bounded-arithmetic-fixture',updates=3,warmup_updates=0,scheduler='constant',
        query_count=8,query_chunk=5,validation_query_count=8,validation_every=500,checkpoint_every=500,**kw)

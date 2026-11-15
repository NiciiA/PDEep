from pathlib import Path
import json,time
import pytest,torch
from pdeep.learning_diagnostics.certificates import resolve_certification

EVIDENCE={}
@pytest.fixture(scope='session',autouse=True)
def resources():
    old=torch.get_num_threads();torch.set_num_threads(1);start=time.perf_counter();yield;torch.set_num_threads(old)
    p=Path(__file__).resolve().parents[2]/'reports/g1_bayesian/bounded_checks.json';p.parent.mkdir(parents=True,exist_ok=True)
    old=json.loads(p.read_text()) if p.exists() else {}
    prior=old.get('measurements',{});prior.update(EVIDENCE)
    p.write_text(json.dumps({'scope':'bounded controls, no research training or certification','seconds':time.perf_counter()-start,'measurements':prior},indent=2)+'\n')

@pytest.fixture(scope='session')
def certificate():
    r=resolve_certification(requested_capabilities={'certification_id':None})
    if not r.verified:pytest.skip('existing native certificate unavailable; no automatic certification')
    return r.require()

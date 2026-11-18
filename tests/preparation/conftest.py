from pathlib import Path
import pytest,torch
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.unified.config import DatasetConfig
from pdeep.preparation.engine import prepare_dataset

@pytest.fixture(scope='session',autouse=True)
def bounded_threads():
    old=torch.get_num_threads();torch.set_num_threads(1);yield;torch.set_num_threads(old)

@pytest.fixture(scope='session')
def certificate():
    result=resolve_certification(requested_capabilities={'certification_id':None})
    if not result.verified:pytest.skip('existing native certificate unavailable; never recertify')
    return result.require()

@pytest.fixture(scope='session')
def prepared(certificate,tmp_path_factory):
    # Eight real production-P4 cases only; one original independent finer check.
    # Four TRAIN cases are required to preserve the independent effective batch.
    config=DatasetConfig(preset='smoke',pdes=('poisson',),families=('square',),counts=(4,2,2),stream_updates=3,stream_queries=16,stream_validation_queries=8,stream_grid=4,qc_points=8)
    data=prepare_dataset(config,tmp_path_factory.mktemp('portable-production-control'),certificate,progress=lambda *a,**k:None)
    assert data.index['status']=='complete',data.index['quarantine']
    import json
    p=Path('reports/g1_preparation');p.mkdir(parents=True,exist_ok=True)
    (p/'production_control_profile.json').write_text(json.dumps({'scope':'eight real P4 Poisson square cases; four TRAIN/two validation/two test; original first-case P3 h.01 independent check; sixteen training queries, no research dataset','physical_cases':len(data.index['records']),'profile':data.index['preparation_profile'],'dataset_identity':data.config.identity,'scientific_execution_source':data.config.reference_execution_source,'reference_fidelity_QC':'passed'},indent=2)+'\n')
    return data

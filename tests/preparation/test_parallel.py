from pathlib import Path
import json
import pytest,numpy as np
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.preparation.parallel import prepare_dataset
from pdeep.preparation.resources import budget
from pdeep.unified.config import DatasetConfig,case_index
from pdeep.smooth.data import Case
from pdeep.artifact_io import read_payload
from pdeep.preparation.portable import read_index

def small():return DatasetConfig(preset='smoke',pdes=('poisson','heat','wave'),families=('square','hole'),counts=(4,2,2),stream_updates=3,stream_queries=8,stream_grid=4,qc_points=8)
def keys(c):return [(p,g,'train',case_index(g,1)) for g in c.families for p in c.pdes]

def test_spawn_worker_counts_determinism_resume_and_thread_limits(certificate,tmp_path):
    c=small();selected=[(p,g,'train',case_index(g,i)) for g in c.families for p in c.pdes for i in (1,2)]
    a=prepare_dataset(c,tmp_path/'one',certificate,workers=1,block_cases=2,_selected_keys=selected,_control={'h':.12,'degree':2},progress=lambda *a,**k:None)
    b=prepare_dataset(c,tmp_path/'two',certificate,workers=2,block_cases=2,_selected_keys=selected,_control={'h':.12,'degree':2},progress=lambda *a,**k:None)
    assert set(a.index['records'])==set(b.index['records']) and len(a.index['records'])==12
    for cid in a.index['records']:
        aa=read_payload(a.root/a.index['records'][cid])[1]['labels'];bb=read_payload(b.root/b.index['records'][cid])[1]['labels'];assert set(aa)==set(bb)
        for key in aa:np.testing.assert_array_equal(aa[key],bb[key])
    for profile in b.index['preparation_profile']['worker_profiles'].values():assert set(profile['BLAS_thread_environment'].values())=={'1'}
    before=dict(b.index['records']);again=prepare_dataset(c,b.root,certificate,manifest=b.path,workers=2,block_cases=2,_selected_keys=selected,_control={'h':.12,'degree':2},progress=lambda *a,**k:None);assert again.index['records']==before
    assert not (b.root/'.preparation-coordinator.json').exists()
    with pytest.raises(ValueError,match='control data'):b.require_ready('train')
    with pytest.raises(ValueError,match='promoted'):prepare_dataset(c,b.root,certificate,manifest=b.path,workers=1,block_cases=1,progress=lambda *a,**k:None)

def test_worker_crash_preserves_private_commit_and_exact_resume(certificate,tmp_path):
    c=small();cid=Case('square','poisson','train',case_index('square',1)).case_id
    try:prepare_dataset(c,tmp_path,certificate,workers=1,block_cases=1,_selected_keys=keys(c),_control={'h':.12,'degree':2},_fault_case=cid,progress=lambda *a,**k:None)
    except (RuntimeError,Exception) as error:assert 'broken' in str(error).lower() or 'terminated' in str(error).lower() or 'shutdown' in str(error).lower()
    path=tmp_path/'manifest.json';assert path.exists();_,index=read_index(path);before=set(index['records'])
    recovered=prepare_dataset(c,tmp_path,certificate,manifest=path,workers=1,block_cases=1,_selected_keys=keys(c),_control={'h':.12,'degree':2},progress=lambda *a,**k:None)
    assert before<=set(recovered.index['records']) and len(recovered.index['records'])==6
    assert recovered.index['preparation_profile']['case_metrics'][cid]['reused_completed_case']

def test_worker_ram_budget_rejects_oversubscription(certificate):
    p=budget(certificate,('square','hole'),requested=1,block_cases=1,available_bytes=16*1024**3)
    assert p['workers']==1 and p['threads_per_worker']==1
    with pytest.raises(ValueError):budget(certificate,('square','hole'),requested=1000,available_bytes=4*1024**3)

def test_invalid_parallel_sizes_reject_before_any_reference_work(certificate,tmp_path):
    for arguments in ({'workers':-1},{'block_cases':0},{'max_cases':-1}):
        with pytest.raises(ValueError):prepare_dataset(small(),tmp_path,certificate,**arguments)
    assert not (tmp_path/'manifest.json').exists()

def test_single_coordinator_guard_covers_initial_manifest_creation(certificate,tmp_path):
    import fcntl
    with (tmp_path/'.preparation.lock').open('a+') as owner:
        fcntl.flock(owner.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        with pytest.raises(ValueError,match='active preparation coordinator'):
            prepare_dataset(small(),tmp_path,certificate)
    assert not (tmp_path/'manifest.json').exists()

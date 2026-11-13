"""No-discovery wave workflow; genuine derivative checks elsewhere, no updates here."""
from pathlib import Path
from dataclasses import asdict,replace
import json,time,random,threading
import numpy as np
import pytest
import torch
from pdeep.core import ValidationError
from pdeep.learning_diagnostics import wave_study as ws
from pdeep.learning_diagnostics.wave_progress import observe
from pdeep.study_monitoring import WaveDashboard
from pdeep.artifact_drive import PersistencePolicy
from pdeep.artifact_io import read_manifest,read_payload,write_artifact
from test_optimization_diagnostics import certificate,unit_data
from test_diagnostic_suite import synthetic_runtime

@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    def fail(*a,**kw):pytest.fail('No real optimizer update or recertification permitted')
    for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):monkeypatch.setattr(cls,'step',fail)
    monkeypatch.setattr('pdeep.smooth.qualification.ensure_compatible_certificate',fail)


def plans(certificate,tmp_path,**kw):
    return ws.prepare_wave_experiments(certificate=certificate,root=tmp_path,capacity='verification',updates=2,query_count=32,query_chunk=17,device='cpu',amp_mode='off',**kw)


def no_discovery(monkeypatch,drive):
    original_glob=Path.glob;original_rglob=Path.rglob
    def guard(fn):
        def wrapped(path,*a,**kw):
            if path.resolve().is_relative_to(drive.resolve()):pytest.fail('Drive directory enumeration is forbidden')
            return fn(path,*a,**kw)
        return wrapped
    monkeypatch.setattr(Path,'glob',guard(original_glob));monkeypatch.setattr(Path,'rglob',guard(original_rglob))
    for name in ('recover_wave_checkpoints','recover_wave_experiments','_snapshot'):
        monkeypatch.setattr(ws,name,lambda *a,**kw:pytest.fail('default wave path must not call recovery/snapshot discovery'))


def test_default_prepare_preview_and_fresh_training_never_discover(certificate,tmp_path,synthetic_runtime,monkeypatch,capsys):
    drive=tmp_path/'drive';drive.mkdir();no_discovery(monkeypatch,drive)
    policy=PersistencePolicy(True,str(drive),True,backend='filesystem_test')
    ps=plans(certificate,tmp_path/'local',persistence=policy)
    assert all(p.wave_lookup=='none' and p.action=='fresh' and not p.wave_resume_paths for p in ps)
    assert ws.run_wave_study(ps,None,notebook=False)['optimizer_updates']==0
    assert 'PHASE G — WAVE TRAINING' in capsys.readouterr().out
    verified={p.model:{'passed':True,'plan_identity':p.identity,'source':p.source} for p in ps}
    result=ws.run_wave_study(ps,verified,run_mode='train',notebook=False)
    assert result['session_optimizer_updates']==10 # synthetic events, no optimizer
    assert all(r['suite_sync']['Drive_verified'] for r in result['architectures'].values())
    before={p.suite_id:Path(result['architectures'][p.model]['snapshot']).read_bytes() for p in ps}
    first_ids=tuple(p.suite_id for p in ps)
    again=ws.run_wave_study(ps,verified,run_mode='train',notebook=False)
    assert again['session_optimizer_updates']==10 and tuple(p.suite_id for p in ps)!=first_ids
    for arch,old in result['architectures'].items():assert Path(old['snapshot']).read_bytes()==before[old['suite_id']]
    assert len(synthetic_runtime.updates)==20


def test_fresh_ids_unique_debug_selection_no_optimizer(certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('preview optimizer'))
    a=plans(certificate,tmp_path);b=plans(certificate,tmp_path)
    assert set(p.suite_id for p in a).isdisjoint(p.suite_id for p in b)
    subset=ws.prepare_wave_experiments(certificate=certificate,root=tmp_path,capacity='verification',updates=2,query_count=32,query_chunk=17,
        architectures=['A_GEO'],experiments={'A_GEO':['BASELINE'],'C_GEO':['KIN_001']},device='cpu')
    assert len(subset)==1 and subset[0].wave_force_fp32
    for kw in ({'action':'resume'},{'checkpoint_lookup':'search'},{'resume_paths':{'A_GEO/BASELINE':'bad'}}):
        with pytest.raises(ValidationError):plans(certificate,tmp_path,**kw)


def test_explicit_paths_only_missing_rejected_without_reference(certificate,tmp_path,monkeypatch):
    monkeypatch.setattr(Path,'glob',lambda *a,**kw:pytest.fail('no glob in explicit path resolution'))
    monkeypatch.setattr(Path,'rglob',lambda *a,**kw:pytest.fail('no recursive lookup'))
    with pytest.raises(ValidationError,match='does not exist'):
        plans(certificate,tmp_path,checkpoint_lookup='explicit',resume_paths={'A_GEO/BASELINE':str(tmp_path/'missing.json')})
    with pytest.raises(ValidationError,match='explicit paths for every'):
        plans(certificate,tmp_path,checkpoint_lookup='explicit',action='resume')


def test_explicit_suite_resume_completed_skipped_no_other_run_reads(certificate,tmp_path,synthetic_runtime,monkeypatch):
    ps=plans(certificate,tmp_path);v={p.model:{'passed':True,'plan_identity':p.identity,'source':p.source} for p in ps}
    r=ws.run_wave_study(ps,v,run_mode='train',notebook=False)
    specified={arch:record['snapshot'] for arch,record in r['architectures'].items()}
    monkeypatch.setattr(ws,'_validate_resume_state',lambda *a,**kw:None) # synthetic payload fixture only
    with monkeypatch.context() as scoped:
        glob=Path.glob;rglob=Path.rglob
        def guard(fn):
            def call(path,*a,**kw):
                if path.is_relative_to(tmp_path):pytest.fail('no saved-run enumeration during explicit prepare')
                return fn(path,*a,**kw)
            return call
        scoped.setattr(Path,'glob',guard(glob));scoped.setattr(Path,'rglob',guard(rglob))
        again=plans(certificate,tmp_path,checkpoint_lookup='explicit',action='resume',resume_paths=specified)
    assert all(p.action=='resume' for p in again)
    before=list(synthetic_runtime.updates)
    rr=ws.run_wave_study(again,v,run_mode='resume',notebook=False)
    assert rr['session_optimizer_updates']==0 and synthetic_runtime.updates==before


def test_heartbeat_preparation_is_display_only_and_fallback_flushes(certificate,tmp_path,monkeypatch,capsys):
    ps=plans(certificate,tmp_path);m=WaveDashboard(ps,notebook=False,heartbeat_seconds=.02)
    called=[];original=m._emit
    def emit(*a,**kw):called.append(threading.current_thread().name);return original(*a,**kw)
    monkeypatch.setattr(m,'_emit',emit)
    random_state=random.getstate();numpy_state=np.random.get_state();torch_state=torch.random.get_rng_state().clone()
    with m.live():
        m.phase('REFERENCE ASSEMBLY',architecture='A_GEO',variant='BASELINE')
        time.sleep(.09)
    assert called.count('wave-display-heartbeat')>=2 and not m._heartbeat.is_alive()
    assert random.getstate()==random_state and np.array_equal(np.random.get_state()[1],numpy_state[1]) and torch.equal(torch_state,torch.random.get_rng_state())
    assert 'REFERENCE ASSEMBLY' in m.last_compact_text and 'Updates: 0/2 successful' in m.last_compact_text
    def fail(*a):raise RuntimeError('display unavailable')
    f=WaveDashboard(ps,notebook=True,renderer=fail)
    f._display('{}',None)
    assert 'PHASE G — WAVE TRAINING' in capsys.readouterr().out and f._fallback


def test_observed_failure_names_stage_no_optimizer(certificate,tmp_path,synthetic_runtime,monkeypatch,capsys):
    ps=plans(certificate,tmp_path);v={p.model:{'passed':True,'plan_identity':p.identity,'source':p.source} for p in ps}
    def fail(*a,**kw):
        return observe('CPU64 REFERENCE TRAJECTORY',lambda:(_ for _ in ()).throw(RuntimeError('bounded test failure')))
    monkeypatch.setattr('pdeep.learning_diagnostics.suite.prepare_diagnostic',fail)
    with pytest.raises(RuntimeError) as caught:ws.run_wave_study(ps,v,run_mode='train',notebook=False)
    text=capsys.readouterr().out
    assert 'CPU64 REFERENCE TRAJECTORY' in text and 'last successful optimizer update: 0' in text
    assert caught.value.wave_result['status']=='failed' and not synthetic_runtime.updates


def test_notebook_exact_four_logical_cells_none_lookup():
    import nbformat,ast
    nb=nbformat.read(Path(__file__).resolve().parents[1]/'notebooks/02_train.ipynb',4)
    cells=[c for c in nb.cells if 'wave_kinematic' in c.metadata.get('tags',[])]
    assert len(cells)==4
    assert 'WAVE_CHECKPOINT_LOOKUP = "none"' in cells[0].source and 'WAVE_ACTION = "fresh"' in cells[0].source
    assert 'WAVE_RESUME_PATHS = {}' in cells[0].source and 'WAVE_ARCHITECTURES' in cells[0].source
    assert 'recover_wave' not in ''.join(c.source for c in cells)
    assert 'WAVE_RESULT = run_wave_study(WAVE_PLAN, WAVE_VERIFICATION' in cells[2].source
    assert 'run_wave_study(' not in cells[3].source
    for c in cells:ast.parse(c.source)
    import inspect
    source=inspect.getsource(ws)
    assert '.rglob(' not in source and '.glob(' not in source and '_candidate_paths' not in source


def test_display_forces_real_success_counters_every_ten_steps(certificate,tmp_path):
    from pdeep.train_types import TrainingState
    ps=ws.prepare_wave_experiments(certificate=certificate,root=tmp_path,updates=30,device='cpu')
    c=ws.member_config(ps[0],'display-only-data','BASELINE','display-events')
    state=TrainingState(c.architecture,c.run_id);state.phase='ready'
    m=WaveDashboard(ps,notebook=False,interval=1e6)
    m.suite_state={'entries':{'wave_single':{'runs':{'BASELINE':{'run_id':c.run_id}},'status':'running'}}}
    from types import SimpleNamespace
    step=SimpleNamespace(unweighted={'field':.1,'auxiliary':.2,'kinematic':.3},weighted={'field':.1,'auxiliary':.02,'kinematic':.0003},
        total=.1203,learning_rate=3e-4,gradient_before_clip=2.,gradient_after_clip=1.,memory={},timings={})
    m.performance={'clipped_fraction':.5,'step_wall_seconds':.25,'queries_per_second':640.,'memory':{'allocated_bytes':123000000}}
    for i in range(31):
        state.successful=i;state.elapsed_seconds=i*.25
        m.update(c,state,step)
    assert m.render_count>=4 and m.last_successful==30
    assert 'Updates: 30/30 successful' in m.last_compact_text
    assert 'Field: 0.1' in m.last_compact_text and 'Kinematic: 0.3' in m.last_compact_text
    assert '50.00%' in m.last_compact_text and 'Queries/s: 640' in m.last_compact_text


def test_genuine_existing_checkpoint_state_validated_without_rng_or_optimizer_mutation(monkeypatch):
    from test_numpy_checkpoint_compat import existing_study
    from pdeep.learning_diagnostics.engine import DiagnosticConfig
    from pdeep.artifact_tensors import tree_identity
    from pdeep.core import digest
    import copy
    checkpoint=existing_study();meta,p=read_payload(checkpoint)
    c=DiagnosticConfig(**{**meta['config'],**{k:tuple(meta['config'][k]) for k in ('pdes','probabilities','training_families')}})
    monkeypatch.setattr(torch.optim.AdamW,'__init__',lambda *a,**kw:pytest.fail('validator must not construct optimizer'))
    before=torch.random.get_rng_state().clone();identity=digest(tree_identity(p))
    ws._validate_resume_state(meta,p,c,meta['execution_qualification'])
    assert torch.equal(before,torch.random.get_rng_state()) and identity==digest(tree_identity(p))
    bad=copy.deepcopy(p);bad['optimizer']['param_groups'][0]['params']=bad['optimizer']['param_groups'][0]['params'][:-1]
    with pytest.raises(ValidationError,match='association'):ws._validate_resume_state(meta,bad,c,meta['execution_qualification'])


def test_fresh_evaluation_directories_do_not_clobber_previous_results(certificate,tmp_path,monkeypatch):
    from pdeep.learning_diagnostics import wave_comparison as comparison
    a=plans(certificate,tmp_path);b=plans(certificate,tmp_path)
    gate={p.model:{} for p in a}
    def report(plan,verification,paths,runs,recovery,root,**kw):return {'directory':str(root),'paths':paths}
    monkeypatch.setattr(comparison,'_evaluate_comparison',report)
    monkeypatch.setattr(ws,'recover_wave_checkpoints',lambda *a,**kw:pytest.fail('no evaluation discovery'))
    first=comparison.compare_wave_experiments(a,gate,display=False)
    second=comparison.compare_wave_experiments(b,gate,display=False)
    assert first['directory']!=second['directory'] and all(v is None for v in first['paths'].values())

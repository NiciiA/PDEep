"""Read-only inspection and isolated tiny scheduling verification; no Colab worker."""
from pathlib import Path
import json
from types import SimpleNamespace
import pytest
import torch
from pdeep.artifact_io import atomic_json,write_artifact
from pdeep.study_monitoring import VariantDashboard,inspect_study,selected_variants,run_study

ORDER=['FIELD_ONLY','WEAK_001','ALL_001','ALL_01','FULL_01']


def state(budget=10):
    runs={v:{'run_id':'run-'+v,'status':'pending','last_successful_update':0,
             'preflight':{'profiles':[{'step_wall_seconds':2. if v=='FIELD_ONLY' else 4.}]}} for v in ORDER}
    return {'suite_id':'synthetic-status','configuration_hash':'configuration','order':['c_physics_ablation'],
        'status':'running','active_experiment':'c_physics_ablation',
        'entries':{'c_physics_ablation':{'status':'running','runs':runs}},
        'preview':{'settings':{'updates':budget},'experiments':[{'experiment':'c_physics_ablation','variants':ORDER}]}}


def test_inspect_saved_sequence_and_concurrently_written_log_readonly(tmp_path):
    s=state(1000);r=s['entries']['c_physics_ablation']['runs']['FIELD_ONLY']
    directory=tmp_path/'original-run';(directory/'checkpoints').mkdir(parents=True)
    r.update(checkpoint=str(directory/'checkpoints/last.json'),last_successful_update=20,observed_update=20)
    (directory/'steps.jsonl').write_text(json.dumps({'run_id':r['run_id'],'successful':25,'status':'successful'})+'\n'+ '{"unfinished":')
    manifest=write_artifact(tmp_path/'snapshots','suite-state-00000001',
        {'artifact_type':'diagnostic_bundle','architecture_id':'C_GEO','run_id':s['suite_id'],'suite_schema':'G1-smooth-diagnostic-suite-v1','configuration_hash':s['configuration_hash']},
        {'suite_state':s,'files':{}},required=('suite_state','files'))
    atomic_json(tmp_path/'latest.json',{'manifest':str(manifest)})
    before={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}
    report=inspect_study(tmp_path)
    assert report['selected_variants']==ORDER # Serialized dict keys sort differently!
    assert report['rows'][0]['status']=='running (inferred)' and report['rows'][0]['successful']==25
    assert report['rows'][0]['checkpointed']==20
    assert [r['status'] for r in report['rows'][1:]]==['pending']*4
    assert all(r['planned']==1000 for r in report['rows'])
    assert before=={p:(p.stat().st_size,p.stat().st_mtime_ns) for p in tmp_path.rglob('*') if p.is_file()}


def test_monitor_variant_totals_ETA_transitions_and_one_handle():
    s=state();calls=[];handle=object()
    d=VariantDashboard(suite_state=s,renderer=lambda text,h:calls.append((json.loads(text.split('\nRecent events:')[0]),h)) or handle)
    d.render({'phase':'preparing','performance':{}})
    assert d.progress['variant_total']==5 and d.progress['variant_index'] is None
    for step in range(3):d.render({'run_id':'run-FIELD_ONLY','successful':step,'elapsed_seconds':2.*step,'performance':{}})
    assert d.progress['label']=='Variant 1/5: FIELD_ONLY'
    assert d.progress['current_variant_updates']==2 and d.progress['current_variant_planned_updates']==10
    assert d.progress['current_variant_ETA_seconds']==16.
    assert d.progress['whole_study_ETA_seconds']==176.
    s['entries']['c_physics_ablation']['runs']['FIELD_ONLY']['status']='completed'
    d.render({'run_id':'run-WEAK_001','successful':1,'elapsed_seconds':4.,'performance':{}})
    assert d.progress['label']=='Variant 2/5: WEAK_001'
    assert d.progress['current_variant_ETA_seconds']==36.
    assert d.progress['whole_study_ETA_seconds']==156.
    assert calls[0][1] is None and all(old is handle for _,old in calls[1:])
    assert calls[-1][0]['experiment']=='Variant 2/5: WEAK_001'
    for run in s['entries']['c_physics_ablation']['runs'].values():run['status']='completed'
    s['status']='completed';s['active_experiment']=None
    d.render({'run_id':'run-FULL_01','successful':10,'elapsed_seconds':40.,'performance':{}})
    assert d.progress['label']=='Study complete: 5/5 variants' and d.progress['whole_study_ETA_seconds']==0.


def test_completed_tail_counts_and_unknown_ETA_are_honest():
    s=state();s['entries']['c_physics_ablation']['runs']['FULL_01']['status']='completed'
    d=VariantDashboard(suite_state=s,notebook=False)
    d.render({'run_id':'run-FIELD_ONLY','successful':0,'elapsed_seconds':0.,'performance':{}})
    assert d.progress['whole_study_successful_updates']==10
    assert d.progress['whole_study_ETA_seconds'] is None
    assert d.progress['current_variant_ETA_seconds'] is None

@pytest.mark.parametrize('raises',[False,True])
def test_future_wrapper_only_supplies_monitor_and_restores_binding(monkeypatch,raises):
    from pdeep.learning_diagnostics import optimization_studies as existing
    calls=[]
    def runner(plan,**kw):calls.append(kw);return 'same-result'
    def study(plan,**kw):
        result=existing.run_diagnostic_suite(plan,expected_identity='same',run_mode='train')
        if raises:raise RuntimeError('test error')
        return result
    monkeypatch.setattr(existing,'run_diagnostic_suite',runner);monkeypatch.setattr(existing,'run_study',study)
    if raises:
        with pytest.raises(RuntimeError,match='test error'):run_study(object(),run_mode='train',notebook=False)
    else:assert run_study(object(),run_mode='train',notebook=False)=='same-result'
    assert len(calls)==1 and isinstance(calls[0]['monitor'],VariantDashboard)
    assert calls[0]['expected_identity']=='same' and existing.run_diagnostic_suite is runner


def test_real_tiny_five_variants_execute_once_in_order(tmp_path,monkeypatch):
    """Five CPU optimizer steps total, verification preset, NEW tmp artifacts only."""
    from pdeep.learning_diagnostics.certificates import resolve_certification
    from pdeep.learning_diagnostics.optimization_studies import prepare_study
    from pdeep.learning_diagnostics.suite import run_diagnostic_suite
    certificate=resolve_certification(requested_capabilities={'certification_id':None})
    if not certificate.verified:pytest.skip('Existing native certificate unavailable; never requalify automatically')
    monkeypatch.setattr('pdeep.smooth.qualification.ensure_compatible_certificate',lambda *a,**kw:pytest.fail('No numerical requalification'))
    plan=prepare_study('c_wave',certificate=certificate.require(),root=tmp_path,capacity='verification',device='cpu',amp_mode='off',query_count=32,query_chunk=16,updates=1,
        advanced={'grid_size':4,'validation_every':1,'checkpoint_every':1})
    renders=[];monitor=VariantDashboard(suite_state={},notebook=False,interval=0,renderer=lambda text,h:renders.append(text) or (h if h is not None else object()))
    original=torch.optim.AdamW.step;steps=[]
    def bounded(opt,*args,**kwargs):
        assert len(steps)<5,'strict five-step total budget'
        steps.append(monitor.progress['current_variant']);return original(opt,*args,**kwargs)
    monkeypatch.setattr(torch.optim.AdamW,'step',bounded)
    result=run_diagnostic_suite(plan,expected_identity=plan.identity,run_mode='train',monitor=monitor)
    assert steps==ORDER
    entry=result['entries']['c_physics_ablation']
    assert all(entry['runs'][v]['last_successful_update']==1 and entry['runs'][v]['status']=='completed' for v in ORDER)
    labels=[json.loads(t.split('\nRecent events:')[0])['experiment'] for t in renders]
    for i,v in enumerate(ORDER,1):assert f'Variant {i}/5: {v}' in labels
    assert monitor.progress['whole_study_planned_updates']==5 and result['status']=='completed'
    # Verify resume skips every completed arm, preserving the five-step budget.
    resumed=prepare_study('c_wave',certificate=certificate.require(),root=tmp_path,capacity='verification',device='cpu',amp_mode='off',query_count=32,query_chunk=16,updates=1,
        advanced={'grid_size':4,'validation_every':1,'checkpoint_every':1})
    again=run_diagnostic_suite(resumed,expected_identity=resumed.identity,run_mode='resume',monitor=VariantDashboard(suite_state={},notebook=False))
    assert again['status']=='completed' and steps==ORDER

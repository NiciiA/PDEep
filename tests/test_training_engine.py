"""Engine arithmetic uses tiny injected models; science adapters use real models.

No test steps a real A/C/B optimizer. The sole actual3-update verification for
all four variants is the notebook preflight, honoring the Stage7A budget.
"""
from dataclasses import asdict,replace
from pathlib import Path
import io,json
import numpy as np
import pytest
import torch
from pdeep.train_types import ResolvedTrainConfig,LossReport,StepReport
from pdeep.train_registry import registration,ScientificAdapter
from pdeep.train_data import TrainingSampler,make_validation_manifest
from pdeep.train_validation import validate,validate_D,model_identity
from pdeep.train_monitor import LiveMonitor,monitor_snapshot
from pdeep.training import TrainingEngine,TrainingFailure,SuccessfulScheduler
from pdeep.train_checkpoint import load_checkpoint
from pdeep.data import calibrate
from pdeep.models import select_architecture
from pdeep.encoding import EquationVocabulary
from pdeep.reference import canonical_spec
from pdeep.b_artifacts import tree_identity,tree_cpu


@pytest.fixture(scope='module')
def norm():return calibrate()


def cfg(name='A',**kwargs):
    values=dict(architecture=name,microbatch=2,accumulation=2,effective_batch=3,query_count=8,validation_query_count=8,source_id='testing-engine',run_tag='arithmetic')
    values.update(kwargs);return ResolvedTrainConfig(**values)


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__();self.weight=torch.nn.Parameter(torch.tensor(.3))
        self.vocabulary=EquationVocabulary.from_specs(tuple(canonical_spec(p) for p in ('poisson','heat','wave')))


class TinyAdapter:
    def loss(self,model,batch,*,detailed=False):
        loss=((model.weight-batch.unknowns.xi[:,0])**2).mean()
        return LossReport(loss,{'toy':loss},{'toy':loss},{'toy':'dimensionless arithmetic test only'},
                          {'cases':len(batch.specs),'valid_queries':int(batch.queries.valid.sum()),'active_coefficients':int(batch.unknowns.active.sum()),'valid_observations':int(batch.observations.valid.sum())})


def tiny_engine(tmp_path,norm,**kwargs):
    return TrainingEngine(cfg(**kwargs),norm,tmp_path,model=TinyModel(),adapter=TinyAdapter(),probe=False)


def test_registry_and_configuration_presets_capability_gates():
    assert set(registration(k).identity for k in ('A','C','B0','B1','D'))=={'A','C','B0','B1','D'}
    for name in ('A','B0','B1'):assert (ResolvedTrainConfig(name,source_id='test').microbatch,ResolvedTrainConfig(name,source_id='test').effective_batch)==(8,16)
    assert (ResolvedTrainConfig('C',source_id='test').microbatch,ResolvedTrainConfig('C',source_id='test').accumulation)==(1,16)
    assert ResolvedTrainConfig('A',preset='research',source_id='test').warmup_updates==250
    for kwargs in ({'architecture':'D'},{'architecture':'B'},{'architecture':'A','precision':'float64'},{'architecture':'A','inverse_track':'T2'},{'architecture':'A','capacity':'small'},{'architecture':'A','effective_batch':17},{'architecture':'A','probabilities':(.2,.2,.2)}):
        with pytest.raises(ValueError):ResolvedTrainConfig(**kwargs)
    assert not registration('D').trainable


@pytest.mark.parametrize('name',['A','C','B0','B1'])
def test_exact_objective_adapter_mapping_no_real_optimizer_updates(norm,name):
    config=cfg(name);batch=TrainingSampler(config,norm).batch(0)[0]
    model=select_architecture(name,normalizers=norm)
    report=ScientificAdapter(name).loss(model,batch)
    if name=='A':
        from pdeep.a_training import objective
        expected=objective(model(batch.specs,batch.observations,batch.queries),batch)
    elif name=='C':
        from pdeep.c_training import objective
        expected=objective(model,model(batch.specs,batch.observations,batch.queries),batch)
    else:
        from pdeep.b_training import coefficient_loss
        expected={'total':coefficient_loss(model.infer_unknowns(model.prepare_context(batch.specs,batch.observations)),batch.unknowns)}
    torch.testing.assert_close(report.total,expected['total'],rtol=0,atol=0)
    torch.testing.assert_close(report.total,sum(report.weighted.values()),rtol=1e-7,atol=1e-7)
    assert set(report.unweighted)==({'field','xi'} if name=='A' else {'field','xi','weak','initial'} if name=='C' else {'xi'})


def test_unequal_microbatch_weighting_counters_and_single_clip(norm,tmp_path,monkeypatch):
    engine=tiny_engine(tmp_path,norm,grad_clip_norm=100.)
    batch,_=engine.sampler.batch(0);expected=2*(.3-batch.unknowns.xi[:,0].mean())
    clip=torch.nn.utils.clip_grad_norm_;calls=[]
    def counted(*args,**kwargs):calls.append(1);return clip(*args,**kwargs)
    monkeypatch.setattr(torch.nn.utils,'clip_grad_norm_',counted)
    r=engine.update(batch)
    torch.testing.assert_close(engine.model.weight.grad,expected,rtol=1e-6,atol=1e-7)
    assert len(calls)==1 and r.successful==r.attempted==1 and r.skipped==r.failed==0
    assert r.counts['cases']==3 and engine.state.microbatches_processed==2 and engine.state.queries_processed==24
    assert not engine.state.accumulation_active and engine.scheduler.completed==1


def test_skip_does_not_tick_success_or_scheduler(norm,tmp_path,monkeypatch):
    engine=tiny_engine(tmp_path,norm,scheduler='warmup_then_cosine',warmup_updates=1)
    before=engine.model.weight.detach().clone();lr=engine.optimizer.param_groups[0]['lr']
    monkeypatch.setattr(engine,'_optimizer_step',lambda:False)
    r=engine.update()
    assert r.status=='skipped' and engine.state.skipped==engine.state.attempted==1
    assert engine.state.successful==engine.scheduler.completed==0 and engine.optimizer.param_groups[0]['lr']==lr
    assert torch.equal(engine.model.weight,before)


def test_scheduler_exact_successful_ordinals_and_restart():
    model=TinyModel();config=cfg(scheduler='warmup_then_cosine',warmup_updates=2,updates=3)
    opt=torch.optim.AdamW(model.parameters(),lr=config.learning_rate);s=SuccessfulScheduler(opt,config)
    assert opt.param_groups[0]['lr']==pytest.approx(1.5e-4)
    s.advance(1);assert opt.param_groups[0]['lr']==pytest.approx(3e-4)
    s.advance(2);assert opt.param_groups[0]['lr']==pytest.approx(3e-5)
    state=s.state_dict();restored=SuccessfulScheduler(opt,config);restored.load_state_dict(state)
    assert restored.completed==2
    with pytest.raises(ValueError):restored.advance(4)


@pytest.mark.parametrize('kind',['loss','gradient','oom','unsupported'])
def test_failure_packet_stops_without_scheduler_advance(norm,tmp_path,kind):
    engine=tiny_engine(tmp_path,norm)
    before=engine.model.weight.detach().clone()
    if kind=='loss':
        class Bad(TinyAdapter):
            def loss(self,*args,**kwargs):
                r=super().loss(*args,**kwargs);r.unweighted['first_bad']=r.total*float('nan');r.total=r.unweighted['first_bad'];return r
        engine.adapter=Bad()
    elif kind=='gradient':engine.model.weight.register_hook(lambda g:g*float('nan'))
    else:
        class Bad(TinyAdapter):
            def loss(self,*args,**kwargs):raise torch.OutOfMemoryError('deliberate mock OOM') if kind=='oom' else NotImplementedError('deliberate unsupported test operation')
        engine.adapter=Bad()
    with pytest.raises(TrainingFailure) as error:engine.update()
    packet=json.loads(Path(error.value.diagnostic).read_text())
    assert packet['case_ids'] and packet['config']['architecture']=='A' and packet['inputs']['value']['dtype']=='torch.float32'
    assert engine.state.failed==engine.state.attempted==1 and engine.state.successful==engine.scheduler.completed==0
    assert torch.equal(before,engine.model.weight)
    if kind=='loss':assert packet['first_component']=='first_bad' and packet['current_loss_components']['first_bad']=='nan'
    if kind=='gradient':assert packet['invalid_gradients']==['weight']
    assert not error.value.step.finite
    with pytest.raises(ValueError):engine.update()
    with pytest.raises(ValueError):engine.save_checkpoint()


def test_fixed_validation_sampler_no_rng_mode_or_gradient_mutation(norm):
    config=cfg('C');sampler=TrainingSampler(config,norm);manifest=make_validation_manifest(config,norm)
    assert all(k[0]=='validation' for k in manifest.keys)
    assert manifest.identity==make_validation_manifest(config,norm).identity
    c=select_architecture('C',normalizers=norm);c.train();c.field.layers[0].eval()
    modes=[m.training for m in c.modules()]
    for p in c.parameters():p.grad=torch.full_like(p,.123)
    gradients=[p.grad.clone() for p in c.parameters()];identity=model_identity(c)
    torch_before=torch.random.get_rng_state().clone();np_before=np.random.get_state();next_keys=sampler.keys(7)
    with torch.inference_mode():report=validate(c,'C',manifest,norm)
    assert model_identity(c)==identity and [m.training for m in c.modules()]==modes
    assert all(torch.equal(p.grad,g) for p,g in zip(c.parameters(),gradients))
    assert torch.equal(torch_before,torch.random.get_rng_state()) and np.array_equal(np_before[1],np.random.get_state()[1]) and sampler.keys(7)==next_keys
    assert set(report.metrics['wave']['quantities'])=={'0','1'}
    assert all('weak_objective' in x for x in report.architecture_metrics['per_case'])
    assert report.case_set_identity==manifest.identity


def test_validation_failure_still_restores_mode(norm):
    config=cfg();manifest=make_validation_manifest(config,norm);m=select_architecture('A',normalizers=norm);m.train()
    class Broken:
        identity='broken';name='broken'
        def batches(self,*args,**kwargs):raise RuntimeError('diagnostic failure')
    with pytest.raises(RuntimeError,match='diagnostic failure'):validate(m,'A',Broken(),norm)
    assert m.training
    d=validate_D(manifest,norm)
    assert d.architecture=='D' and len(d.architecture_metrics['posterior_uncertainty'])==3


def test_live_monitor_expected_fields_and_eta(norm,tmp_path):
    stream=io.StringIO();monitor=LiveMonitor(stream=stream,suite_budgets={'A':3,'C':3})
    engine=TrainingEngine(cfg(),norm,tmp_path,model=TinyModel(),adapter=TinyAdapter(),probe=False,monitor=monitor)
    step=engine.update();snapshot=monitor.latest
    for key in ('run_id','source_id','config_id','phase','unweighted','weighted','timings','memory','checkpoint','Drive_sync','learning_rate','gradient_before_clip','cases_per_second','suite_ETA_seconds'):assert key in snapshot
    assert snapshot['ETA_seconds'] is None and snapshot['suite_ETA_seconds'] is None
    assert 'objective' in stream.getvalue() and 'float32' in stream.getvalue()
    engine.state.recent_steps=[asdict(step)]*20
    assert monitor_snapshot(engine.config,engine.state,step)['ETA_seconds'] is not None


def test_independent_parameters_and_optimizer_state(norm,tmp_path):
    a=tiny_engine(tmp_path/'one',norm);c=tiny_engine(tmp_path/'two',norm,name='C')
    a.update();c.update()
    assert a.optimizer is not c.optimizer and a.model.weight.data_ptr()!=c.model.weight.data_ptr()
    assert a.optimizer.state[a.model.weight]['exp_avg'].data_ptr()!=c.optimizer.state[c.model.weight]['exp_avg'].data_ptr()


def test_checkpoint_completed_boundary_immutable_resume_arithmetic(norm,tmp_path,monkeypatch):
    engine=tiny_engine(tmp_path,norm)
    with pytest.raises(ValueError):engine.save_checkpoint()
    engine.update();path=engine.save_checkpoint()
    with pytest.raises(ValueError):engine.save_checkpoint()
    from pdeep import models,train_registry
    monkeypatch.setattr(models,'select_architecture',lambda *args,**kwargs:TinyModel())
    # Reload still uses the same explicit objective injected for arithmetic tests.
    monkeypatch.setattr('pdeep.training.ScientificAdapter',lambda *args:TinyAdapter())
    restored=load_checkpoint(path,expected_architecture='A',probe=False)
    assert restored.state.successful==restored.scheduler.completed==1 and restored.state.sampler_cursor==engine.state.sampler_cursor
    assert tree_identity(tree_cpu(restored.optimizer.state_dict()))==tree_identity(tree_cpu(engine.optimizer.state_dict()))
    engine.update();restored.update()
    torch.testing.assert_close(engine.model.weight,restored.model.weight,rtol=0,atol=0)
    with pytest.raises(ValueError,match='architecture'):load_checkpoint(path,expected_architecture='C')
    payload=path.with_suffix('.pt');payload.write_bytes(payload.read_bytes()+b'bad')
    with pytest.raises(ValueError,match='checksum'):load_checkpoint(path,expected_architecture='A')


def test_full_training_requires_both_deliberate_controls(norm,tmp_path):
    config=cfg(preset='research',updates=10000)
    engine=TrainingEngine(config,norm,tmp_path,model=TinyModel(),adapter=TinyAdapter(),probe=False)
    with pytest.raises(ValueError,match='full training disabled'):engine.run()
    with pytest.raises(ValueError,match='full training disabled'):engine.run(launch_full=True)


@pytest.mark.parametrize('name',['A','C','B0','B1'])
@pytest.mark.parametrize('device',['cpu','mps','cuda'])
def test_actual_capability_and_fixed_validation_no_optimizer_updates(norm,tmp_path,name,device):
    from pdeep.runtime import capabilities
    if not next(c.available for c in capabilities() if c.device==device):pytest.skip(device+' unavailable')
    c=cfg(name,device=device,run_tag='native-'+name)
    engine=TrainingEngine(c,norm,tmp_path)
    assert engine.capability['passed'] and engine.state.successful==0
    before=torch.random.get_rng_state().clone();opt=tree_identity(tree_cpu(engine.optimizer.state_dict()));identity=model_identity(engine.model)
    mode=engine.model.training;gradients={n:None if p.grad is None else p.grad.clone() for n,p in engine.model.named_parameters()}
    report=engine.validate()
    assert report.counts['cases']==3 and report.device==device
    assert engine.model.training==mode and model_identity(engine.model)==identity and tree_identity(tree_cpu(engine.optimizer.state_dict()))==opt
    assert torch.equal(before,torch.random.get_rng_state())
    assert all(p.grad is None if gradients[n] is None else torch.equal(p.grad,gradients[n]) for n,p in engine.model.named_parameters())


@pytest.mark.parametrize('name',['A','C','B0','B1'])
def test_adapter_padding_and_inactive_targets_are_excluded(norm,name):
    from pdeep.representation_checks import padded
    config=cfg(name);batch=TrainingSampler(config,norm).batch(0)[0];model=select_architecture(name,normalizers=norm);adapter=ScientificAdapter(name)
    original=adapter.loss(model,batch)
    obs=padded(batch.observations,4);queries=padded(batch.queries,3)
    target=torch.cat((batch.target,torch.full((3,3,1),float('nan'))),dim=1)
    unknown=replace(batch.unknowns,xi=torch.where(batch.unknowns.active,batch.unknowns.xi,torch.full_like(batch.unknowns.xi,float('nan'))))
    modified=replace(batch,observations=obs,queries=queries,target=target,unknowns=unknown)
    actual=adapter.loss(model,modified)
    torch.testing.assert_close(original.total,actual.total,rtol=2e-5,atol=2e-6)
    assert original.counts==actual.counts


def test_unsupported_probe_fails_before_optimizer_and_persists_startup(norm,tmp_path,monkeypatch):
    def unsupported(*args,**kwargs):raise NotImplementedError('mock unsupported coordinate derivative')
    def optimizer_forbidden(*args,**kwargs):raise AssertionError('optimizer should not initialize')
    monkeypatch.setattr('pdeep.training.capability_probe',unsupported)
    monkeypatch.setattr(torch.optim,'AdamW',optimizer_forbidden)
    config=cfg(run_tag='startup-failure')
    with pytest.raises(NotImplementedError,match='coordinate derivative'):
        TrainingEngine(config,norm,tmp_path,model=TinyModel())
    packet=json.loads((tmp_path/config.architecture/config.run_id/'startup_failure.json').read_text())
    assert packet['kind']=='startup_capability_failure' and packet['state']['attempted']==0


def test_validation_frequency_and_plotting_do_not_advance_training_order(norm):
    from pdeep.diagnostics import RandomStreams
    one=TrainingSampler(cfg(validation_every=1),norm);two=TrainingSampler(cfg(validation_every=250),norm)
    before=torch.random.get_rng_state().clone()
    for i in range(5):
        assert one.keys(i)==two.keys(i)
        RandomStreams(one.config.root_seed).numpy('monitoring','extra-plot',i).normal(size=100)
    assert torch.equal(before,torch.random.get_rng_state())

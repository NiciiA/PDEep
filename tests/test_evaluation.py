"""Bounded independent Stage 7B tests; no scientific model optimization."""
from dataclasses import replace, asdict, FrozenInstanceError
from pathlib import Path
import json
import subprocess
import sys
import pytest
import torch
from pdeep.core import ValidationError, digest
from pdeep.data import calibrate, CaseDataset, observations, array_hash
from pdeep.reference import canonical_spec, evaluate, prior_scales
from pdeep.bayes import ExactGaussianD, condition_covariance, observed_system, modal_prior
from pdeep.models import select_architecture
from pdeep.batches import ObservationBatch
from pdeep.evaluation_plan import make_plan, EvaluationPlan
from pdeep.evaluation import EvaluationConfig, evaluate as evaluate_plan
from pdeep.evaluation_metrics import errors, aggregate, coverage, paired_interval, coefficient_errors, coverage_interval
from pdeep.evaluation_inference import Method, slice_query
from pdeep.evaluation_physics import physical_diagnostics, diagnostic_query
from pdeep.evaluation_diagnostics import (input_identity, observation_variants, observation_diagnostics, ComparisonProtocol, equation_structure_hook)
from pdeep.representation_checks import padded
from pdeep.train_validation import model_identity

@pytest.fixture(scope='module')
def norm():return calibrate()

@pytest.fixture(scope='module')
def config():return EvaluationConfig()

@pytest.fixture(scope='module')
def plan(norm,config):return make_plan(norm,source_id='test-source',config_id=config.identity)

@pytest.fixture(scope='module')
def methods(norm):
    return [Method(n,n,'untrained',select_architecture(n,normalizers=norm)) for n in ('A','C','B0','B1')]+[Method('D','D','exact; no training')]


def test_plan_roundtrip_fresh_process_and_tampering(plan,tmp_path):
    path=plan.save(tmp_path);other=EvaluationPlan.load(path);assert plan==other and plan.identity==other.identity
    code='from pdeep.evaluation_plan import EvaluationPlan;import sys;print(EvaluationPlan.load(sys.argv[1]).identity)'
    assert subprocess.check_output([sys.executable,'-c',code,str(path)],text=True).strip()==plan.identity
    with pytest.raises(FrozenInstanceError):plan.split='test'
    data=json.loads(path.read_text());data['plan']['cases'][0]['case_id']=2;path.write_text(json.dumps(data))
    with pytest.raises(ValidationError,match='checksum'):EvaluationPlan.load(path)


def test_plan_splits_no_test_without_explicit_authorization(norm,config,plan,tmp_path):
    with pytest.raises(ValidationError):make_plan(norm,source_id='x',config_id=config.identity,split='test')
    test=make_plan(norm,source_id='x',config_id=config.identity,split='test',allow_final_test=True)
    assert not {c.physical_id for c in test.cases}&{c.physical_id for c in plan.cases}
    path=test.save(tmp_path)
    with pytest.raises(ValidationError):EvaluationPlan.load(path)
    with pytest.raises(ValidationError):evaluate_plan(test,[Method('D','D','exact; no training')],norm,config=config)
    assert EvaluationPlan.load(path,allow_final_test=True)==test
    with pytest.raises(ValidationError):replace(plan,cases=(replace(plan.cases[0],physical_id='poisson/00000'),)+plan.cases[1:]).validate()
    with pytest.raises(ValidationError):replace(plan,spatial_points=plan.spatial_points[:-1]).validate()


def test_canonical_grid_axes_and_regeneration(plan,norm):
    for entry in plan.cases:
        case,cond,held=plan.regenerate(entry,norm)
        assert cond.metadata.identity!=held.metadata.identity and cond.metadata.case_identity==held.metadata.case_identity==case.identity
        for qid in dict(plan.quantities)[entry.pde]:
            q=plan.query(entry.pde,qid);assert q.position.dtype==torch.float64 and q.position.device.type=='cpu'
            assert len(q.position[0])==(4096 if entry.pde=='poisson' else 9*4096)
            assert bool(q.time_present.any())==(entry.pde!='poisson')
            assert set(q.quantity_id.flatten().tolist())=={qid}
    with pytest.raises(ValidationError):plan.query('poisson',1)


def metric(value,count=1):return {'physical_MSE':value,'physical_MAE':value,'standardized_MSE':value,'count':count}

def test_independent_reduction_known_answer():
    rows=[{'pde':'poisson','case_id':'p1','quantities':{'u':metric(2,1000)}},
          {'pde':'poisson','case_id':'p2','quantities':{'u':metric(4,1)}},
          {'pde':'heat','case_id':'h','quantities':{'u':metric(9,17)}},
          {'pde':'wave','case_id':'w','quantities':{'u':metric(1,100000),'v':metric(25,1)}}]
    r=aggregate(rows)
    assert r['by_PDE']['wave']['equal_quantity_metrics']['standardized_MSE']==13
    assert r['by_PDE']['wave']['quantities']['v']['physical_RMSE']==5
    assert r['macro']['standardized_MSE']==pytest.approx(25/3)
    assert r['macro']['physical_RMSE']==pytest.approx((25/3)**.5)
    p=torch.tensor([[[2.],[4.],[float('nan')]]]);t=torch.zeros_like(p);mask=torch.tensor([[True,True,False]])
    score=errors(p,t,2.,mask);assert score['physical_MSE']==10 and score['standardized_MSE']==2.5 and score['physical_MAE']==3


def test_case_bootstrap_not_points():
    rows=[{'pde':p,'case_id':str(i),'quantities':{'u':metric(float(i+1),99999)}} for p in ('heat','wave') for i in range(3)]
    shifted=[{**r,'quantities':{'u':metric(r['quantities']['u']['standardized_MSE']+2)}} for r in rows]
    ci=paired_interval(rows,shifted,seed=2)
    assert ci['estimate']==-2 and ci['CI95']==[-2,-2]
    with pytest.raises(ValidationError):paired_interval(rows,shifted[::-1])
    assert paired_interval(rows[:1],shifted[:1])['CI95'] is None
    cov=coverage_interval([{'pde':'wave','coverage':{'0.95':.9}}])
    assert cov['0.95']['CI95']==[0,1]

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_D_mean_noise_padding_and_independent_posterior(norm,plan,pde):
    case=next(c for c in plan.cases if c.pde==pde);truth,cond,held=plan.regenerate(case,norm);s=canonical_spec(pde)
    m=Method('D','D','exact; no training');session=m.prepare(s,cond.inputs);post=session.context
    H,y,sigma=observed_system(s,cond.inputs);prior=modal_prior(s)
    independent=condition_covariance(H,y,sigma,prior.mean,prior.covariance)
    assert torch.allclose(post.mean,independent.mean,rtol=2e-10,atol=1e-12)
    assert torch.allclose(post.covariance,independent.covariance,rtol=1e-9,atol=2e-15)
    q=diagnostic_query(plan,s);pred=session.predict(q,chunk=7)
    assert torch.equal(pred,post.predict(q).mean)
    result=post.predict(held.inputs,sigma=held.inputs.sigma)
    assert torch.allclose(result.variance,result.latent_variance+held.inputs.sigma.square(),rtol=0,atol=1e-17)
    pad=m.prepare(s,padded(cond.inputs,5))
    assert torch.equal(pad.xi,session.xi)
    bank=post.sample(23,generator=torch.Generator().manual_seed(42))
    assert torch.allclose(bank.evaluate(q),torch.cat([bank.evaluate(slice_query(q,i,i+1)) for i in range(len(plan.diagnostic_points))],1),rtol=32*torch.finfo(torch.float64).eps,atol=32*torch.finfo(torch.float64).eps)
    # A deliberately changed bank cannot affect the posterior-mean Session API.
    assert not torch.equal(pred,bank.evaluate(q)[:1])
    with pytest.raises(ValidationError):m.prepare(s,replace(cond.inputs,sigma=torch.zeros_like(cond.inputs.sigma)))

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_actual_modal_AD_physics_signs_initial_and_boundary(norm,plan,pde):
    entry=next(c for c in plan.cases if c.pde==pde);case,cond,_=plan.regenerate(entry,norm)
    session=Method('D','D','exact; no training').prepare(canonical_spec(pde),cond.inputs)
    result=physical_diagnostics(session,plan,case.tensor(),norm)
    scale=1+float(session.xi.abs().max())
    assert max(result['max_abs_residual'])<2e-11*scale,result
    assert max(result['independent_AD_vs_analytic_max'].values())<2e-11*scale
    assert max(result['boundary_max_abs'].values())<2e-13*scale
    if pde!='poisson':
        assert result['initial']['u']['inferred_initial_consistency']['physical_MSE']<1e-27
        assert result['initial']['u']['hidden_truth_error']['physical_MSE']>0
    assert len(result['max_abs_residual'])==(2 if pde=='wave' else 1)

@pytest.mark.parametrize('index',range(5))
def test_all_methods_chunk_padding_context_and_no_training(norm,plan,methods,index):
    method=methods[index];entry=plan.cases[2];case,cond,_=plan.regenerate(entry,norm);spec=canonical_spec('wave')
    before=None if method.model is None else model_identity(method.model)
    grad_before=None if method.model is None else [p.grad for p in method.model.parameters()]
    state=method.prepare(spec,cond.inputs);q=diagnostic_query(plan,spec)
    full=state.predict(q,chunk=24);chunk=state.predict(q,chunk=7)
    single=torch.cat([state.predict(slice_query(q,i,i+1),chunk=1) for i in range(24)],1)
    assert torch.allclose(full,chunk,rtol=2e-5,atol=2e-7)
    assert torch.allclose(full,single,rtol=2e-5,atol=2e-7)
    padded_state=method.prepare(spec,padded(cond.inputs,7))
    assert torch.allclose(full,padded_state.predict(q),rtol=2e-5,atol=2e-7)
    if method.model is not None:
        assert model_identity(method.model)==before
        assert [p.grad for p in method.model.parameters()]==grad_before
    assert method.status in ('untrained','exact; no training')
    assert state.method is method
    if method.architecture.startswith('B'):assert len(state.context.trajectory[0])==4


def test_contexts_owned_by_separate_neural_instances(norm,plan,methods):
    _,cond,_=plan.regenerate(plan.cases[0],norm);spec=canonical_spec('poisson');q=diagnostic_query(plan,spec)
    from pdeep.representations import float32_batch
    for method in methods[:4]:
        other=select_architecture(method.architecture,normalizers=norm)
        state=method.prepare(spec,cond.inputs)
        with pytest.raises(ValidationError,match='model'):
            other.predict_queries(state.context,float32_batch(q))
    assert not {p.data_ptr() for p in methods[2].model.parameters()} & {p.data_ptr() for p in methods[3].model.parameters()}


def test_observation_controls_all_cases_independent_unchanged(norm,plan):
    m=Method('D','D','exact; no training')
    for entry in plan.cases:
        case,cond,_=plan.regenerate(entry,norm);original=input_identity(cond.inputs);spec=canonical_spec(entry.pde)
        r=observation_diagnostics(m,spec,cond.inputs,diagnostic_query(plan,spec))['experiments']
        assert input_identity(cond.inputs)==original
        assert r['record_permutation']['field_max_change']<1e-12
        assert r['value_reassignment']['field_max_change']>1e-6
        assert r['change_first_value']['field_max_change']>1e-7
        assert r['sigma_x2']['covariance_difference_min_eigenvalue']>-1e-14
        assert r['remove_even_records']['covariance_difference_min_eigenvalue']>-1e-14
        variants=observation_variants(cond.inputs,entry.pde)
        empty=m.prepare(spec,variants['empty'][0]);assert torch.equal(empty.xi,modal_prior(spec).mean)
        assert torch.equal(empty.context.covariance,modal_prior(spec).covariance)
        for name in ('move_first_sensor','change_first_time','flip_first_quantity'):
            if name in variants:
                assert torch.equal(variants[name][0].value,cond.inputs.value)
                assert torch.equal(variants[name][0].sigma,cond.inputs.sigma)


def test_coverage_latent_measurement_and_coefficients():
    m=torch.zeros(4,dtype=torch.float64);v=torch.ones_like(m);t=torch.tensor([0.,.7,1.7,3.],dtype=torch.float64)
    assert coverage(m,v,t)=={'0.5':.25,'0.9':.5,'0.95':.75}
    assert coverage(m,4*v,t)['0.95']==1
    out=coefficient_errors(torch.ones(18),torch.zeros(18),torch.ones(18)*2)
    assert out['physical_RMSE']==1 and out['prior_standardized_MSE']==.25 and len(out['per_mode_error'])==18


def test_protocols_do_not_train_and_structural_hooks(norm,plan,methods):
    assert ComparisonProtocol().validate(serious=True)
    assert ComparisonProtocol(protocol='matched_wall_clock',successful_updates=None,wall_clock_seconds=60).validate(serious=True)
    with pytest.raises(ValidationError):ComparisonProtocol(seeds=(1,)).validate(serious=True)
    with pytest.raises(ValidationError):ComparisonProtocol(equation_encoding='label_only').validate()
    spec=canonical_spec('heat');changed=replace(spec,bindings=tuple(replace(b,value=.04) if b.id=='alpha' else b for b in spec.bindings))
    for method in methods[:4]:
        normal=equation_structure_hook(method,spec,None);different=equation_structure_hook(method,changed,None)
        assert normal['accepted'] and different['accepted']
        assert normal['context_identity']!=different['context_identity'] and not different['physical_solver_supported']
    assert not equation_structure_hook(methods[-1],changed,None)['accepted']


def test_matched_runner_reproducibility_no_hidden_truth_or_model_mutation(norm,methods,monkeypatch,tmp_path):
    c=EvaluationConfig(observation_diagnostics=False,physical_diagnostics=False)
    p=make_plan(norm,source_id='test',config_id=c.identity)
    seen=[];original=Method.prepare
    def checked(self,spec,obs):
        assert isinstance(obs,ObservationBatch) and not hasattr(obs,'metadata') and not hasattr(obs,'xi')
        seen.append((self.name,spec.name,input_identity(obs)))
        return original(self,spec,obs)
    monkeypatch.setattr(Method,'prepare',checked)
    selected=methods
    before=torch.random.get_rng_state().clone(); identities={m.name:model_identity(m.model) for m in methods if m.model}
    result=evaluate_plan(p,selected,norm,config=c,output=tmp_path)
    assert torch.equal(before,torch.random.get_rng_state())
    assert result['scientific']['optimizer_steps']==0 and not result['scientific']['test_evaluated']
    assert result['scientific']['methods']['A']['training_status']=='untrained'
    for entry in p.cases:
        assert len({identity for name,pde,identity in seen if pde==entry.pde})==1
    assert set(result['scientific']['aggregate']['D']['by_PDE']['wave']['quantities'])=={'u','v'}
    assert identities=={m.name:model_identity(m.model) for m in methods if m.model}
    again=evaluate_plan(p,selected,norm,config=c)
    assert result['identity']==again['identity']
    assert result['scientific']==again['scientific']
    assert result['performance']!=again['performance']

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_strong_diagnostic_nonzero_manufactured_known_answer(plan,pde):
    from types import SimpleNamespace
    from pdeep.constraints import strong_diagnostics
    class Manufactured:
        def evaluate_state(self,context,q):
            x=q.position[...,0:1];y=q.position[...,1:2];t=q.physical_time
            return SimpleNamespace(u=x*x+y*y+t*t*t,v=2*t+x*y)
        def infer_unknowns(self,context):return SimpleNamespace(xi_hat=torch.zeros(1,18,dtype=torch.float64))
    spec=canonical_spec(pde);q=diagnostic_query(plan,spec);t=q.physical_time[0,:,0];x=q.position[0,:,0];y=q.position[0,:,1]
    report=strong_diagnostics(Manufactured(),SimpleNamespace(specs=(spec,)),q)[0]
    expected=[torch.full_like(t,-4)] if pde=='poisson' else [3*t*t-.08] if pde=='heat' else [3*t*t-2*t-x*y,torch.full_like(t,.04)]
    for observed,answer in zip(report['residual_values'],expected):assert torch.allclose(observed,answer,rtol=2e-13,atol=2e-13)


def test_B_disabled_spatial_no_cached_features(norm,plan,methods):
    from pdeep.architecture_b import BAblation
    spec=canonical_spec('wave');_,cond,_=plan.regenerate(plan.cases[2],norm);q=diagnostic_query(plan,spec)
    b1=methods[3];control=Method('B1-no-spatial','B1','untrained',b1.model,ablation=BAblation(spatial=False))
    first=control.prepare(spec,cond.inputs)
    _=b1.prepare(spec,cond.inputs)
    second=control.prepare(spec,cond.inputs)
    assert len(first.context.spatial_history)==0 and len(second.context.spatial_history)==0
    assert torch.equal(first.xi,second.xi) and torch.equal(first.predict(q),second.predict(q))
    assert methods[2].name=='B0' and b1.name=='B1'

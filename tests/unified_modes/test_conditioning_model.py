from dataclasses import replace
import numpy as np
import pytest,torch
from pdeep.smooth.data import Case
from pdeep.unified.data import Provider
from pdeep.unified_modes.conditioning import FieldCondition,CaseView,mask_for
from pdeep.unified_modes.evaluation import information_view,empty_observations
from pdeep.unified_modes.model import create
from .conftest import EVIDENCE

@pytest.mark.parametrize('pde',('poisson','heat','wave'))
@pytest.mark.parametrize('family',('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon'))
def test_all24_information_semantics_and_forward_basis(dataset,pde,family):
    c=Case(family,pde,'train',0);d=64 if pde=='wave' else 32;a=np.concatenate([f.coefficients for f in c.fields.values()]);full=FieldCondition.from_case(c,np.ones(d,bool));hidden=FieldCondition.from_case(c,np.zeros(d,bool))
    assert full.complete and hidden.known_fraction==0 and not hidden.values.any();assert set(hidden.status)=={0,1} and set(full.status)=={0,2}
    secret=np.full(d,np.nan);scrub=FieldCondition.from_coefficients(pde,family,secret,np.zeros(d,bool));assert scrub.identity==hidden.identity
    model=create('A_GEO',dataset.norms,'verification');view=CaseView(c,full,'FORWARD');ot=empty_observations(c,dataset.norms);ctx=model.prepare_context((view.problem,),ot)[0];x=torch.tensor(c.problem.mesh.sample(12,np.random.default_rng(3)),dtype=torch.float32);t=x.new_zeros((len(x),1)) if pde!='poisson' else None
    for role,f in c.fields.items():
        actual=model.evaluate(ctx,x,t,role).detach().double().numpy()[:,0];expected=f(x.double().numpy());np.testing.assert_allclose(actual,expected,atol=3e-7*max(abs(expected).max(),1e-3),rtol=3e-5)
    EVIDENCE.setdefault('24_tasks',{})[pde+'/'+family]='complete/hidden/N-A, sanitized NaN secrets, exact known C3 field evaluation passed'

@pytest.mark.parametrize('architecture',('A_GEO','C_GEO'))
def test_counterfactuals_invariances_and_gradients(dataset,architecture):
    c=Case('hole','wave','train',1);model=create(architecture,dataset.norms,'verification');provider=Provider(dataset);mask=np.zeros(64,bool);mask[:32]=True
    view,ot,_=information_view(c,dataset,'HYBRID',mask,8);qt=provider.token_set(c,8,set_id=('unified-validation',42))[0]
    y=model.predict(model.prepare_context((view.problem,),ot),qt,chunk=3)
    def run(f,obs=ot,mode='HYBRID',chunk=8):return model.predict(model.prepare_context((CaseView(c,f,mode).problem,),obs),qt,chunk=chunk)
    f=view.condition;changed_known=f.values.copy();changed_known[np.flatnonzero(f.available)[0]]+=.7;f2=replace(f,values=changed_known);z=run(f2);delta=float((z-y).detach().abs().max());assert delta>1e-7
    perm=np.random.default_rng(1).permutation(96);np.testing.assert_allclose(run(f.permute(perm)).detach(),y.detach(),rtol=3e-5,atol=1e-7)
    np.testing.assert_allclose(run(f,mode='FORWARD').detach(),y.detach(),rtol=2e-5,atol=1e-7) # Label is reporting-only; information is identical.
    order=torch.randperm(8);ops=type(ot)(**{k:None if v is None else v[:,order] for k,v in vars(ot).items()});np.testing.assert_allclose(run(f,ops).detach(),y.detach(),rtol=2e-5,atol=1e-7)
    changed=type(ot)(**{k:None if v is None else v.clone() for k,v in vars(ot).items()});changed.value+=.3;obs_delta=float((run(f,changed)-y).detach().abs().max());assert obs_delta>1e-7
    y.square().mean().backward();assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.known_field_encoder.parameters());assert any(p.grad is not None and p.grad.abs().max()>0 for p in model.observation.parameters())
    EVIDENCE['counterfactual/'+architecture]={'known_component_max_change':delta,'observation_max_change':obs_delta,'interpretation':'untrained response/gradient reachability; not learned forward accuracy'}


def test_forward_empty_padding_and_query_truth_invariance(dataset):
    c=Case('square','poisson','train',0);model=create('C_GEO',dataset.norms,'verification');f=FieldCondition.from_case(c,np.ones(32,bool));p=CaseView(c,f,'FORWARD').problem;obs=empty_observations(c,dataset.norms);qt=Provider(dataset).token_set(c,8,set_id=('unified-validation',42))[0]
    y=model.predict(model.prepare_context((p,),obs),qt,chunk=1)
    pad=Provider(dataset).token_set(c,8,observation=True)[0];pad.valid[:]=False;pad.value[:]=float('nan');pad.physical_time[:]=777
    z=model.predict(model.prepare_context((p,),pad),qt,chunk=8);torch.testing.assert_close(y,z,atol=1e-8,rtol=1e-5)
    qt.value[:]=float('nan');torch.testing.assert_close(z,model.predict(model.prepare_context((p,),obs),qt),atol=1e-8,rtol=1e-5)


def test_structured_wave_known_roles_and_unknown_secret_invariance(dataset):
    c=Case('annulus','wave','train',0);model=create('C_GEO',dataset.norms,'verification');x=torch.tensor(c.problem.mesh.sample(8,np.random.default_rng(17)),dtype=torch.float32);t=x.new_zeros((8,1));obs=empty_observations(c,dataset.norms)
    for role in ('u0','v0'):
        mask,_=mask_for('wave','HYBRID',np.random.default_rng(1),pattern=role);condition=FieldCondition.from_case(c,mask);view=CaseView(c,condition,'HYBRID');ctx=model.prepare_context((view.problem,),obs)[0]
        np.testing.assert_allclose(model.evaluate(ctx,x,t,role).detach().numpy()[:,0],c.fields[role](x.double().numpy()),rtol=3e-5,atol=3e-8)
        a=np.concatenate([c.fields[r].coefficients for r in ('u0','v0')]);a[~mask]=np.nan;hidden_changed=FieldCondition.from_coefficients('wave','annulus',a,mask);assert hidden_changed.identity==condition.identity

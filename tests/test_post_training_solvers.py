"""Separate post-training classical controls: no neural optimization."""
import numpy as np
import pytest
import torch
from scipy.sparse.linalg import cg
from pdeep.post_training_solvers import Quadratic,gradient_descent,conjugate_gradient,BUDGETS
from pdeep.post_training_benchmark import observation_quadratic,score
from pdeep.bayes import ExactGaussianD
from pdeep.b_physics import build_system
from pdeep.core import ValidationError
from pdeep.data import CaseDataset,calibrate,observations
from pdeep.reference import canonical_spec

@pytest.fixture(scope='module')
def norm():return calibrate()

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_same_objective_gradient_CG_independent_residual_and_D(norm,pde):
    ds=CaseDataset(norm.root_seed);case=ds.case('validation',pde,21);spec=canonical_spec(pde)
    obs=observations(ds,case,norm,set_id='post-training-check').inputs;q=observation_quadratic(spec,obs)
    system=build_system((spec,),obs);initial=q.scale*np.random.default_rng(43).normal(size=len(q.b))
    x=torch.tensor(np.pad(initial,(0,18-len(initial)))[None],dtype=torch.float64,requires_grad=True)
    _,_,analytic,phi=system.state(x);ad=torch.autograd.grad(phi.sum(),x)[0]
    np.testing.assert_allclose(q.independent_gradient(initial),analytic[0,:len(initial)].detach().numpy(),atol=1e-9,rtol=1e-12)
    np.testing.assert_allclose(q.P@initial-q.b,ad[0,:len(initial)].numpy(),atol=1e-9,rtol=1e-12)
    assert q.phi(initial)==pytest.approx(float(phi.detach()),rel=1e-14)
    solution=conjugate_gradient(q,initial,trace=True);other,info=cg(q.P,q.b,x0=initial,rtol=1e-12,atol=1e-12)
    D=ExactGaussianD().fit(spec,obs)
    assert solution['converged'] and info==0 and len(solution['trajectory'])==solution['iterations']+1
    np.testing.assert_allclose(solution['xi'],other,atol=1e-9,rtol=1e-9)
    np.testing.assert_allclose(solution['xi'],D.mean.numpy(),atol=1e-9,rtol=1e-9)
    independent=q.independent_gradient(solution['xi'])
    assert np.linalg.norm(independent)<=solution['residual_threshold']+1e-9
    assert solution['counts']['precision_matvecs']>=solution['iterations']+2
    assert solution['counts']['true_residual_recomputations']>=1
    for scaling in ('identity','jacobi','prior'):
        answer=gradient_descent(q,initial,300,scaling=scaling,trace=True)
        phi=np.array([q.phi(xi) for xi in answer['trajectory']])
        assert np.all(np.diff(phi)<=1e-9*(1+phi[:-1]))
        assert answer['counts']['gradient_evaluations']==answer['counts']['precision_matvecs']==300
        assert answer['iterations']==300 and answer['counts']['objective_evaluations']==0
        assert np.all(answer['diagonal']>0)
        if scaling=='jacobi':
            np.testing.assert_allclose(answer['diagonal'],1/np.diag(q.P),rtol=0,atol=0)
            assert not np.allclose(np.diag(answer['diagonal'])@q.P,np.eye(len(q.b))) # not a full inverse


def test_declared_gradient_step_one_iteration_and_empty_prior():
    q=Quadratic.build(np.array([[1.,2.],[3.,1.]]),np.array([1.,-.5]),np.array([.2,.4]),np.zeros(2),np.array([.3,.5]))
    initial=np.array([.7,-.4]);r=gradient_descent(q,initial,1,scaling='jacobi')
    np.testing.assert_allclose(r['xi'],initial-r['eta']*r['diagonal']*q.independent_gradient(initial),rtol=1e-14)
    empty=Quadratic.build(np.zeros((0,2)),np.zeros(0),np.zeros(0),np.array([.2,.5]),np.array([.1,.4]))
    result=conjugate_gradient(empty,np.zeros(2))
    np.testing.assert_allclose(result['xi'],empty.mean,atol=1e-13)
    immediate=conjugate_gradient(empty,empty.mean)
    assert immediate['converged'] and immediate['iterations']==0
    for invalid in (0.,-1.):
        with pytest.raises(ValidationError):Quadratic.build(np.eye(2),np.ones(2),np.array([invalid,1.]),np.zeros(2),np.ones(2))


def test_score_gap_not_subtractive_and_active_prior_standardization():
    q=Quadratic.build(np.eye(2),np.ones(2),np.ones(2),np.zeros(2),np.array([.5,1.]))
    D=np.linalg.solve(q.P,q.b);truth=np.array([.4,.1]);xi=D+np.array([.2,-.3])
    result=score(q,xi,truth,D,{'u':(np.eye(2),2.)})
    assert result['optimality_gap']==pytest.approx(.5*(xi-D)@q.P@(xi-D))
    assert result['coefficient_standardized_MSE']==pytest.approx(np.mean(((xi-truth)/q.scale)**2))
    assert result['field_standardized_MSE']==pytest.approx(np.sum((xi-truth)**2)/4)

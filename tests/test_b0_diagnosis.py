"""Independent inverse-objective audit tests; no trained artifact or training needed."""
from dataclasses import replace
import numpy as np
import pytest
import torch
from pdeep.b0_diagnosis import independent_matrix,independent_scales,classical_optimum
from pdeep.b_physics import build_system
from pdeep.bayes import ExactGaussianD,sensor_matrix
from pdeep.data import CaseDataset,calibrate,observations
from pdeep.reference import canonical_spec

@pytest.fixture(scope='module')
def norm():return calibrate()

@pytest.mark.parametrize('pde',['poisson','heat','wave'])
def test_independent_operator_prior_gradient_and_classical_optimum(pde,norm):
    ds=CaseDataset(norm.root_seed);case=ds.case('validation',pde,17)
    obs=observations(ds,case,norm,set_id='diagnosis-independent-check').inputs;spec=canonical_spec(pde)
    H=sensor_matrix(spec,obs).numpy();sd=independent_scales(pde);d=len(sd)
    expected=independent_matrix(pde,obs.position[0].numpy(),obs.physical_time[0].numpy(),obs.quantity_id[0].numpy())
    np.testing.assert_allclose(H,expected,atol=5e-15,rtol=1e-14)
    system=build_system((spec,),obs)
    np.testing.assert_array_equal(sd,system.scale[0,:d].numpy())
    initial=sd*np.random.default_rng(715).normal(size=d)*5
    optimum,info,J,target=classical_optimum(H,obs.value[0,:,0].numpy(),obs.sigma[0,:,0].numpy(),np.zeros(d),sd,initial)
    posterior=ExactGaussianD().fit(spec,obs)
    assert info['success'] and info['nfev']<=20
    np.testing.assert_allclose(optimum,posterior.mean.numpy(),atol=1e-12,rtol=1e-11)
    assert info['vs_augmented_SVD_max_abs']<1e-12
    np.testing.assert_array_equal(info['initial_xi'],initial)
    x=torch.tensor(np.pad(initial,(0,18-d))[None],requires_grad=True)
    _,_,gradient,phi=system.state(x)
    ad=torch.autograd.grad(phi.sum(),x)[0]
    torch.testing.assert_close(ad,gradient,atol=1e-9,rtol=1e-12)
    residual=J@((initial)/sd)-target
    assert float(phi.detach())==pytest.approx(.5*residual@residual,rel=1e-14)
    # Padding is excluded from H, R and the objective, even with invalid payload.
    from pdeep.representation_checks import padded
    padded_system=build_system((spec,),padded(obs,4))
    np.testing.assert_allclose(padded_system.state(x.detach())[-1].numpy(),phi.detach().numpy(),rtol=0,atol=0)
    assert np.all(padded_system.sigma[0,-4:].numpy()==1.)


def test_classical_optimum_empty_observations_returns_prior():
    sd=np.linspace(.1,.9,9);mean=np.arange(9)*.01
    optimum,info,_,_=classical_optimum(np.zeros((0,9)),np.zeros(0),np.zeros(0),mean,sd,mean+2*sd)
    np.testing.assert_allclose(optimum,mean,atol=1e-14,rtol=0)
    assert info['success']

from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from scipy.stats import norm

from pdeep.batches import QueryBatch, UnknownCoefficients
from pdeep.core import ValidationError
from pdeep.data import CaseDataset, case_coefficients, evaluation_grid, collate_inputs
from pdeep.reference import (basis, basis_gradient, canonical_spec, eigenvalues, evaluate,
                             evaluate_batch, prior_log_density, prior_scales, require_supported, sample_prior)
from pdeep.reference_checks import mathematical_checks, gradient


def test_independent_mathematical_checks():
    report = mathematical_checks()
    assert report["passed"], [c for c in report["checks"] if not c["passed"]]
    assert all(c["max_abs_error"] <= c["tolerance"] for c in report["checks"])
    # 3 modal checks + 3 cases each with 4 Poisson, 6 heat, 12 wave checks.
    assert len(report["checks"]) == 3 + 3*(4+6+12)


def test_basis_order_and_orthonormality():
    from numpy.polynomial.legendre import leggauss
    points, weights = leggauss(32)
    x, y = np.meshgrid((points+1)/2,(points+1)/2,indexing="ij")
    phi = basis(torch.from_numpy(np.stack((x.ravel(),y.ravel()),-1)))
    w = torch.from_numpy(np.outer(weights/2,weights/2).ravel())
    gram = phi.T @ (phi*w[:,None])
    torch.testing.assert_close(gram,torch.eye(9,dtype=torch.float64),rtol=1e-12,atol=1e-12)
    expected = torch.tensor([math.pi**2*(m*m+n*n) for m in (1,2,3) for n in (1,2,3)],dtype=torch.float64)
    torch.testing.assert_close(eigenvalues(),expected,rtol=1e-15,atol=1e-14)


@pytest.mark.parametrize("name", ["poisson","heat","wave"])
def test_priors_log_density_and_moments(name):
    scales=prior_scales(name)
    samples=sample_prior(name,np.random.default_rng(149),count=12000)
    assert samples.shape == (12000,18 if name=="wave" else 9)
    z=samples/scales
    assert float(z.mean(0).abs().max()) < .04
    assert float((z.std(0)-1).abs().max()) < .04
    expected=norm.logpdf(samples[:4].numpy(),loc=0.,scale=scales.numpy()).sum(-1)
    np.testing.assert_allclose(prior_log_density(name,samples[:4]).numpy(),expected,rtol=1e-13,atol=1e-13)
    if name=="wave":
        cross = z[:,:9].T @ z[:,9:]/len(z)
        assert float(cross.abs().max()) < .05


@pytest.mark.parametrize("name", ["poisson","heat","wave"])
def test_continuous_queries_coefficient_gradients_and_chunks(name):
    case=CaseDataset().case("train",name,1)
    xi=case.tensor().requires_grad_()
    p=torch.tensor([[.12345,.72723],[.88817,.02131],[.33421,.91231]],dtype=torch.float64)
    t=None if name=="poisson" else torch.tensor([[.01371],[.46437],[.91719]],dtype=torch.float64)
    value=evaluate(name,xi,p,t)
    chunk=torch.cat([evaluate(name,xi,p[i:i+1],None if t is None else t[i:i+1]) for i in range(3)])
    torch.testing.assert_close(value,chunk,rtol=1e-14,atol=1e-14)
    # Independently finite-difference every active modal coefficient.
    derivative=torch.autograd.grad(value.sum(),xi)[0]
    for j in range(len(xi)):
        step=torch.zeros_like(xi); step[j]=1e-5
        finite=(evaluate(name,xi.detach()+step,p,t).sum()-evaluate(name,xi.detach()-step,p,t).sum())/(2e-5)
        torch.testing.assert_close(derivative[j],finite,rtol=1e-9,atol=1e-10)


def test_wave_velocity_spatial_and_time_derivatives():
    xi=CaseDataset().case("train","wave",4).tensor()
    p=torch.tensor([[.23,.47],[.67,.29]],dtype=torch.float64,requires_grad=True)
    t=torch.tensor([[.1234567],[.8765432]],dtype=torch.float64,requires_grad=True)
    v=evaluate("wave",xi,p,t,quantity="v")
    torch.testing.assert_close(gradient(v,t),evaluate("wave",xi,p,t,quantity="v",time_order=1),rtol=1e-12,atol=1e-12)
    torch.testing.assert_close(gradient(gradient(v,t),t),evaluate("wave",xi,p,t,quantity="v",time_order=2),rtol=1e-12,atol=1e-12)
    for j in range(2):
        order=(1,0) if j==0 else (0,1)
        torch.testing.assert_close(gradient(v,p)[:,j:j+1],evaluate("wave",xi,p,t,quantity="v",spatial_derivative=order),rtol=1e-12,atol=1e-12)


def test_stationary_time_precision_and_modified_physics_fail():
    xi=CaseDataset().case("train","poisson",0).tensor()
    p=torch.tensor([[.2,.3]],dtype=torch.float64)
    for arguments in [{"physical_time":torch.zeros(1,1,dtype=torch.float64)}, {"time_order":1}]:
        with pytest.raises(ValidationError,match="no physical-time"):
            evaluate("poisson",xi,p,**arguments)
    with pytest.raises(ValidationError,match="CPU float64"):
        basis(p.float())
    with pytest.raises(ValidationError,match="continuous physical time"):
        evaluate("heat",xi,p)
    spec=canonical_spec("heat")
    changed=replace(spec,bindings=tuple(replace(b,value=.03) if b.id=="alpha" else b for b in spec.bindings))
    with pytest.raises(ValidationError,match="canonical T1"):
        require_supported(changed)


def test_evaluation_grid_quantities_and_padded_batch():
    dataset=CaseDataset()
    cases=tuple(dataset.case("train",p,0) for p in ("poisson","heat","wave"))
    queries=[evaluation_grid(c.key.pde,quantity_id=1 if c.key.pde=="wave" else 0,resolution=4) for c in cases]
    assert queries[0].position.shape==(1,16,2)
    assert queries[1].position.shape==(1,9*16,2)
    q=collate_inputs(queries)
    specs=tuple(canonical_spec(c.key.pde) for c in cases)
    q.validate(specs)
    coefficients=case_coefficients(cases)
    out=evaluate_batch(specs,coefficients,q)
    assert out.shape==(3,144,1)
    assert torch.equal(out[0,16:],torch.zeros(128,1,dtype=torch.float64))
    for i,c in enumerate(cases):
        single=evaluate_batch((specs[i],),case_coefficients((c,)),queries[i])
        torch.testing.assert_close(out[i,:single.shape[1]],single[0],rtol=1e-13,atol=1e-13)
    with pytest.raises(ValidationError,match="no evaluation time"):
        evaluation_grid("poisson",times=(0.,))

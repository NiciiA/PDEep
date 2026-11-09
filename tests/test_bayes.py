from dataclasses import replace
import math
import numpy as np
import pytest
import torch
from scipy.stats import multivariate_normal, norm

from pdeep.batches import ObservationBatch, QueryBatch
from pdeep.bayes import (ExactGaussianD, GaussianPosterior, condition_information,condition_covariance,
                        observed_system,sensor_matrix,select_architecture,regularized_least_squares)
from pdeep.bayes_checks import bayesian_checks, slice_query
from pdeep.core import ValidationError
from pdeep.data import CaseDataset,calibrate,observations,collate_inputs,training_queries,evaluation_grid
from pdeep.diagnostics import RandomStreams
from pdeep.reference import canonical_spec,evaluate


@pytest.fixture(scope="module")
def data():
    return CaseDataset(),calibrate()


def empty_like(obs, *, zero_length=False):
    result=replace(obs,valid=torch.zeros_like(obs.valid))
    if zero_length:
        from dataclasses import fields
        result=type(obs)(**{f.name:getattr(obs,f.name)[:,:0] for f in fields(obs)})
    return result


@pytest.mark.parametrize("name",["poisson","heat","wave"])
def test_analytical_sensor_matrix_against_reference_and_jacobian(data,name):
    d,n=data;c=d.case("validation",name,3);o=observations(d,c,n).inputs;s=canonical_spec(name)
    H=sensor_matrix(s,o)
    t=None if name=="poisson" else o.physical_time[0]
    def forward(xi):
        u=evaluate(name,xi,o.position[0],t)
        if name=="wave":
            u=torch.where(o.quantity_id[0,:,None]==1,evaluate(name,xi,o.position[0],t,quantity="v"),u)
        return u[:,0]
    jacobian=torch.autograd.functional.jacobian(forward,c.tensor())
    torch.testing.assert_close(H,jacobian,rtol=1e-13,atol=1e-13)
    torch.testing.assert_close(H@c.tensor(),forward(c.tensor()),rtol=1e-13,atol=1e-13)


@pytest.mark.parametrize("name",["poisson","heat","wave"])
def test_independent_forms_and_properties(data,name):
    d,n=data;c=d.case("validation",name,0);o=observations(d,c,n).inputs;s=canonical_spec(name)
    p=ExactGaussianD().fit(s,o);H,y,sigma=observed_system(s,o)
    covariance_form=condition_covariance(H[:5],y[:5],sigma[:5],p.prior.mean,p.prior.covariance)
    information_form=condition_information(H[:5],y[:5],sigma[:5],p.prior.mean,p.prior.covariance)
    torch.testing.assert_close(information_form.mean,covariance_form.mean,rtol=1e-10,atol=1e-11)
    torch.testing.assert_close(information_form.covariance,covariance_form.covariance,rtol=1e-10,atol=1e-11)
    assert torch.isfinite(p.mean).all() and torch.isfinite(p.covariance).all()
    torch.testing.assert_close(p.covariance,p.covariance.T,rtol=0,atol=1e-14)
    assert torch.linalg.eigvalsh(p.covariance).min()>0
    assert torch.linalg.eigvalsh(p.prior.covariance-p.covariance).min()>=-1e-12
    assert p.covariance.trace()<p.prior.covariance.trace()
    torch.testing.assert_close(regularized_least_squares(H,y,sigma,p.prior),p.mean,rtol=1e-10,atol=1e-11)


@pytest.mark.parametrize("name",["poisson","heat","wave"])
@pytest.mark.parametrize("zero_length",[True,False])
def test_empty_observations_return_exact_prior(data,name,zero_length):
    d,n=data;c=d.case("train",name,2);o=empty_like(observations(d,c,n).inputs,zero_length=zero_length)
    if not zero_length:
        o=replace(o,position=torch.full_like(o.position,float("nan")),physical_time=torch.full_like(o.physical_time,float("nan")),
                  value=torch.full_like(o.value,float("nan")),sigma=torch.full_like(o.sigma,float("nan")),quantity_id=torch.full_like(o.quantity_id,-99))
    p=ExactGaussianD().fit(canonical_spec(name),o)
    assert torch.equal(p.mean,p.prior.mean) and torch.equal(p.covariance,p.prior.covariance)
    assert p.diagnostics["H"]["rank"]==0
    scores=p.log_predictive_density(o)
    assert scores["joint_log_density"]==scores["sum_marginal_log_density"]==0.


def test_noiseless_negative_sigma_and_nonfinite_fail(data):
    d,n=data;c=d.case("train","heat",0);o=observations(d,c,n).inputs;s=canonical_spec("heat")
    for value in (0.,-1.,float("nan")):
        bad=o.sigma.clone();bad[0,0,0]=value
        with pytest.raises(ValidationError): ExactGaussianD().fit(s,replace(o,sigma=bad))
    with pytest.raises(ValidationError,match="CPU float64"):
        condition_information(torch.ones(1,2),torch.ones(1),torch.ones(1),torch.zeros(2),torch.eye(2))
    with pytest.raises(ValidationError,match="noiseless"):
        condition_information(torch.ones(1,2,dtype=torch.float64),torch.ones(1,dtype=torch.float64),torch.zeros(1,dtype=torch.float64),torch.zeros(2,dtype=torch.float64),torch.eye(2,dtype=torch.float64))


def test_mixed_batch_padding_and_permutation(data):
    d,n=data;names=("poisson","heat","wave")
    records=[observations(d,d.case("validation",p,1),n).inputs for p in names]
    packed=collate_inputs(records)
    from dataclasses import fields
    for i,name in enumerate(names):
        p=ExactGaussianD().fit(canonical_spec(name),packed,row=i)
        single=ExactGaussianD().fit(canonical_spec(name),records[i])
        torch.testing.assert_close(p.mean,single.mean,rtol=1e-12,atol=1e-12)
        order=torch.arange(packed.position.shape[1]-1,-1,-1)
        perm=type(packed)(**{f.name:getattr(packed,f.name)[:,order] for f in fields(packed)})
        after=ExactGaussianD().fit(canonical_spec(name),perm,row=i)
        torch.testing.assert_close(after.mean,p.mean,rtol=1e-11,atol=1e-12)
        torch.testing.assert_close(after.covariance,p.covariance,rtol=1e-11,atol=1e-12)


def test_uninformative_zero_time_wave_leaves_velocity_prior(data):
    d,n=data;c=d.case("train","wave",0);o=observations(d,c,n).inputs
    o=replace(o,physical_time=torch.zeros_like(o.physical_time),quantity_id=torch.zeros_like(o.quantity_id))
    p=ExactGaussianD().fit(canonical_spec("wave"),o)
    torch.testing.assert_close(p.covariance[9:,9:],p.prior.covariance[9:,9:],rtol=0,atol=0)
    assert p.diagnostics["H"]["rank"]==9 and p.diagnostics["H"]["nullity"]==9
    assert p.diagnostics["H"]["column_condition_number"] is None
    assert p.covariance[:9,:9].trace()<p.prior.covariance[:9,:9].trace()


def test_prediction_and_log_scores_independent_scipy(data):
    d,n=data;c=d.case("validation","wave",1);s=canonical_spec("wave")
    o=observations(d,c,n).inputs;h=observations(d,c,n,set_id="heldout",count=8).inputs
    p=ExactGaussianD().fit(s,o);H,y,sigma=observed_system(s,h)
    pred=p.predict(h,sigma=h.sigma)
    C=H@p.covariance@H.T+torch.diag(sigma.square())
    mean=H@p.mean
    torch.testing.assert_close(pred.variance[0,:,0],C.diag(),rtol=1e-12,atol=1e-14)
    assert torch.equal(pred.mean,p.predict(h).mean)
    scores=p.log_predictive_density(h)
    assert scores["joint_log_density"]==pytest.approx(multivariate_normal.logpdf(y.numpy(),mean=mean.numpy(),cov=C.numpy()),rel=1e-11,abs=1e-11)
    np.testing.assert_allclose(scores["marginal"][0,:,0].numpy(),norm.logpdf(y.numpy(),loc=mean.numpy(),scale=C.diag().sqrt().numpy()),rtol=1e-11,atol=1e-11)


def test_coherent_samples_reproducibility_rng_and_wave_relation(data):
    d,n=data;c=d.case("validation","wave",1);s=canonical_spec("wave");o=observations(d,c,n).inputs
    p=ExactGaussianD().fit(s,o);streams=RandomStreams(123)
    before=torch.random.get_rng_state().clone()
    bank=p.sample(4,generator=streams.torch("tests","posterior"))
    again=p.sample(4,generator=streams.torch("tests","posterior"))
    assert torch.equal(bank.xi,again.xi) and torch.equal(before,torch.random.get_rng_state())
    assert not torch.equal(bank.xi[0],p.mean)
    q=training_queries(d,c).inputs
    chunks=torch.cat([bank.evaluate(slice_query(q,0,63)),bank.evaluate(slice_query(q,63,128))],dim=1)
    torch.testing.assert_close(bank.evaluate(q),chunks,rtol=1e-13,atol=1e-13)
    for sample in bank.xi:
        t=q.physical_time[0].detach().requires_grad_();x=q.position[0]
        u=evaluate("wave",sample,x,t)
        dt=torch.autograd.grad(u.sum(),t)[0]
        v=evaluate("wave",sample,x,t,quantity="v")
        torch.testing.assert_close(dt,v,rtol=1e-12,atol=1e-12)


def test_full_bayesian_monte_carlo_checks(data):
    _,normalizers=data
    report=bayesian_checks(normalizers=normalizers)
    assert report["passed"],[c for c in report["checks"] if not c["passed"]]
    assert select_architecture("D").trainable_parameters==0
    for name in ("A","B","C","T2"):
        with pytest.raises(ValidationError,match="unsupported"):select_architecture(name)

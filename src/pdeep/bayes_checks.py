"""Bounded verification and example diagnostics for D; no training."""
from __future__ import annotations

import time
import torch

from .batches import QueryBatch
from .bayes import (ExactGaussianD, coefficient_metrics, condition_covariance,
                    condition_information, observed_system, regularized_least_squares,
                    sensor_matrix)
from .data import CaseDataset, calibrate, observations, training_queries
from .diagnostics import RandomStreams
from .reference import canonical_spec, prior_scales


def slice_query(query,start,stop):
    return QueryBatch(*(getattr(query,key)[:,start:stop] for key in
                        ("position","physical_time","time_present","quantity_id","valid")))


def bayesian_checks(root_seed=20261108, *, draws=20000, normalizers=None):
    require_draws = type(draws) is int and draws >= 1000
    from .core import require
    require(require_draws,"Monte Carlo verification requires at least 1000 draws")
    started = time.perf_counter()
    streams = RandomStreams(root_seed)
    dataset = CaseDataset(root_seed)
    normalizers = calibrate(root_seed) if normalizers is None else normalizers
    checks, examples = [], []
    max_mean, max_covariance = 0.,0.
    def check(name,error,tolerance):
        checks.append({"name":name,"max_error":float(error),"tolerance":float(tolerance),"passed":float(error)<=float(tolerance)})
    # A nonzero mean and correlated prior check general algebra independently.
    g = streams.torch("tests","D","general-correlated-prior")
    L0 = torch.tensor([[.7,0.,0.],[.2,.5,0.],[-.1,.15,.8]],dtype=torch.float64)
    C0 = L0@L0.T
    m0 = torch.tensor([.3,-.5,.1],dtype=torch.float64)
    H = torch.randn(4,3,dtype=torch.float64,generator=g)
    y = torch.randn(4,dtype=torch.float64,generator=g)
    sigma = torch.tensor([.1,.2,.4,.3],dtype=torch.float64)
    a = condition_information(H,y,sigma,m0,C0)
    b = condition_covariance(H,y,sigma,m0,C0)
    check("correlated_prior/mean_forms",(a.mean-b.mean).abs().max(),1e-11)
    check("correlated_prior/covariance_forms",(a.covariance-b.covariance).abs().max(),1e-12)
    for pde in ("poisson","heat","wave"):
        case = dataset.case("validation",pde,0)
        spec = canonical_spec(pde)
        obs = observations(dataset,case,normalizers)
        heldout = observations(dataset,case,normalizers,set_id="heldout")
        fit_start = time.perf_counter()
        posterior = ExactGaussianD().fit(spec,obs.inputs)
        fit_seconds = time.perf_counter()-fit_start
        H,y,sigma = observed_system(spec,obs.inputs)
        check(pde+"/sensor_operator",(H@case.tensor()-obs.metadata.clean[0,:,0]).abs().max(),1e-12)
        # Tiny five-sensor subset for independent physical-covariance route.
        tiny_info = condition_information(H[:5],y[:5],sigma[:5],posterior.prior.mean,posterior.prior.covariance)
        tiny_cov = condition_covariance(H[:5],y[:5],sigma[:5],posterior.prior.mean,posterior.prior.covariance)
        mean_error = float((tiny_info.mean-tiny_cov.mean).abs().max())
        cov_error = float((tiny_info.covariance-tiny_cov.covariance).abs().max())
        max_mean,max_covariance = max(max_mean,mean_error),max(max_covariance,cov_error)
        check(pde+"/mean_forms",mean_error,1e-11)
        check(pde+"/covariance_forms",cov_error,1e-11)
        check(pde+"/symmetry",(posterior.covariance-posterior.covariance.T).abs().max(),1e-14)
        minimum = float(torch.linalg.eigvalsh(posterior.covariance).min())
        check(pde+"/positive_covariance",max(0.,-minimum),0.)
        minimum_contraction = float(torch.linalg.eigvalsh(posterior.prior.covariance-posterior.covariance).min())
        check(pde+"/contraction_psd",max(0.,-minimum_contraction),1e-12)
        require(posterior.covariance.trace()<posterior.prior.covariance.trace(),"informative observations must contract trace")
        bank = posterior.sample(draws,generator=streams.torch("tests","D",pde,"MC"))
        empirical_mean = bank.xi.mean(0)
        centered = bank.xi-empirical_mean
        empirical_covariance = centered.T@centered/(draws-1)
        mean_se = (posterior.covariance.diag()/draws).sqrt()
        covariance_se = ((posterior.covariance.square()+posterior.covariance.diag()[:,None]*posterior.covariance.diag()[None,:])/(draws-1)).sqrt()
        mean_mc_z = float(((empirical_mean-posterior.mean)/mean_se).abs().max())
        cov_mc_z = float(((empirical_covariance-posterior.covariance)/covariance_se).abs().max())
        check(pde+"/MC_mean_standard_errors",mean_mc_z,6.)
        check(pde+"/MC_covariance_standard_errors",cov_mc_z,6.)
        q = training_queries(dataset,case,count=128).inputs
        pred = posterior.predict(q)
        parts = [slice_query(q,0,37),slice_query(q,37,91),slice_query(q,91,128)]
        chunk_mean = torch.cat([posterior.predict(part).mean for part in parts],dim=1)
        check(pde+"/query_mean_chunks",(pred.mean-chunk_mean).abs().max(),1e-12)
        chunk_variance = torch.cat([posterior.predict(part).latent_variance for part in parts],dim=1)
        check(pde+"/query_variance_chunks",(pred.latent_variance-chunk_variance).abs().max(),1e-12)
        small_bank = type(bank)(bank.spec,bank.xi[:8])
        chunk_samples = torch.cat([small_bank.evaluate(part) for part in parts],dim=1)
        sample_chunk_error = float((small_bank.evaluate(q)-chunk_samples).abs().max())
        check(pde+"/coherent_sample_chunks",sample_chunk_error,1e-12)
        ridge = regularized_least_squares(H,y,sigma,posterior.prior,ridge=1.)
        wls = regularized_least_squares(H,y,sigma,posterior.prior,ridge=0.)
        check(pde+"/augmented_LS_MAP",(ridge-posterior.mean).abs().max(),1e-11)
        query_record = training_queries(dataset,case,count=128)
        query_H = sensor_matrix(spec,query_record.inputs)
        target = query_record.target[0,:,0]
        field_errors = {name:float((query_H@estimate-target).square().mean().sqrt())
                        for name,estimate in (("posterior_mean",posterior.mean),("prior_only",posterior.prior.mean),("ridge_1_MAP",ridge),("unregularized_WLS",wls))}
        conditioning_prediction = posterior.predict(obs.inputs,sigma=obs.inputs.sigma)
        heldout_prediction = posterior.predict(heldout.inputs,sigma=heldout.inputs.sigma)
        scale = prior_scales(pde)
        examples.append({"pde":pde,"fit_seconds":fit_seconds,"case_id":case.identity,
                         "conditioning_observation_id":obs.metadata.identity,"heldout_observation_id":heldout.metadata.identity,
                         "diagnostics":posterior.diagnostics,
                         "mean":posterior.mean.tolist(),"covariance":posterior.covariance.tolist(),
                         "posterior_mean_recovery":coefficient_metrics(posterior.mean,case.tensor(),scale),
                         "prior_only_recovery":coefficient_metrics(posterior.prior.mean,case.tensor(),scale),
                         "ridge_1_MAP_recovery":coefficient_metrics(ridge,case.tensor(),scale),
                         "unregularized_WLS_recovery":coefficient_metrics(wls,case.tensor(),scale),
                         "query_field_rmse":field_errors,
                         "conditioning_prediction_mean":conditioning_prediction.mean[0,:,0].tolist(),
                         "conditioning_predictive_variance":conditioning_prediction.variance[0,:,0].tolist(),
                         "heldout_prediction_mean":heldout_prediction.mean[0,:,0].tolist(),
                         "heldout_latent_variance":heldout_prediction.latent_variance[0,:,0].tolist(),
                         "heldout_predictive_variance":heldout_prediction.variance[0,:,0].tolist(),
                         "MC_empirical_mean_max_error":float((empirical_mean-posterior.mean).abs().max()),
                         "MC_empirical_covariance_max_error":float((empirical_covariance-posterior.covariance).abs().max()),
                         "MC_covariance_relative_frobenius":float(torch.linalg.matrix_norm(empirical_covariance-posterior.covariance)/torch.linalg.matrix_norm(posterior.covariance)),
                         "coherent_sample_chunk_max_error":sample_chunk_error,
                         "heldout_scores":{k:v for k,v in posterior.log_predictive_density(heldout.inputs).items() if k!="marginal"},
                         "conditioning_replicate_scores":{k:v for k,v in posterior.log_predictive_density(obs.inputs).items() if k!="marginal"}})
    max_mean = max(max_mean,checks[0]["max_error"])
    max_covariance = max(max_covariance,checks[1]["max_error"])
    return {"passed":all(c["passed"] for c in checks),"device":"cpu","dtype":"float64",
            "draws_per_pde":draws,"max_mean_form_discrepancy":max_mean,
            "max_covariance_form_discrepancy":max_covariance,
            "elapsed_seconds":time.perf_counter()-started,"checks":checks,"examples":examples}

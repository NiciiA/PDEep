"""Stage 7B matched evaluation. Never trains or selects on the test split."""
from dataclasses import dataclass, asdict
from pathlib import Path
import json, math, time
import numpy as np
import torch
from .core import digest, require
from .data import array_hash, case_coefficients, persist_json
from .reference import canonical_spec, evaluate_batch, prior_scales
from .bayes import ExactGaussianD, sensor_matrix, observed_system, condition_covariance
from .evaluation_metrics import errors, coefficient_errors, aggregate, paired_interval, coverage, coverage_interval
from .evaluation_inference import slice_query, sync
from .evaluation_physics import physical_diagnostics, diagnostic_query
from .evaluation_diagnostics import observation_diagnostics, input_identity
from .train_validation import model_identity
from .train_monitor import memory_report

@dataclass(frozen=True)
class EvaluationConfig:
    chunk: int=1024
    observation_diagnostics: bool=True
    physical_diagnostics: bool=True
    posterior_samples: int=64
    warmup: bool=True

    def __post_init__(self):
        require(type(self.chunk) is int and self.chunk>0 and self.posterior_samples>=2,'invalid evaluation budget')
    @property
    def identity(self):return digest(asdict(self))


def sensor_scores(session, observation_set, normalizers, chunk):
    obs=observation_set.inputs;pred=session.predict(obs,chunk=chunk);result={}
    for quantity in ((0,1) if session.spec.name=='wave' else (0,)):
        valid=obs.valid & (obs.quantity_id==quantity)
        if not bool(valid.any()):continue
        s=normalizers.scale(session.spec.name,quantity)
        result['v' if quantity else 'u']={'noisy_residual':errors(pred,obs.value,s,valid),
                                          'clean_field_error':errors(pred,observation_set.metadata.clean,s,valid)}
        e=(pred-obs.value)[valid];sigma=obs.sigma[valid]
        result['v' if quantity else 'u']['noise_standardized_residual_MSE']=float((e/sigma).square().mean())
    return result


def refinement_scores(session, truth, dmean, obs):
    c=session.context;d=len(truth);scale=prior_scales(session.spec.name)
    trajectory=c.trajectory[0,:,:d].detach().cpu().double();phi=c.objectives[0].detach().cpu().double()
    H=sensor_matrix(session.spec,obs)[obs.valid[0]];y=obs.value[obs.valid][...,0];sig=obs.sigma[obs.valid][...,0]
    return {'xi_by_iteration':trajectory.tolist(),'Phi_by_iteration':phi.tolist(),
            'coefficient_vs_truth':[coefficient_errors(x,truth,scale) for x in trajectory],
            'coefficient_vs_D_mean':[coefficient_errors(x,dmean,scale) for x in trajectory],
            'sensor_noise_standardized_MSE':[float(((H@x-y)/sig).square().mean()) for x in trajectory],
            'update_L2':[float(v.norm()) for v in trajectory.diff(dim=0)],
            'Phi_differences':phi.diff().tolist(),'monotone_with_roundoff':bool((phi.diff()<=1e-10*(1+phi[:-1].abs())).all()),
            'spatial_history_steps':len(getattr(c,'spatial_history',()))}


def d_uncertainty(session, case, cond, held, queries, targets, config, seed):
    post=session.context;fields={}; arrays={}
    for name,q in queries.items():
        result=post.predict(q);target=targets[name]
        fields[name]={'latent_variance_mean':float(result.latent_variance.mean()),'latent_coverage':coverage(result.mean,result.latent_variance,target,q.valid)}
        arrays[name]=result.latent_variance.numpy()
    held_pred=post.predict(held.inputs,sigma=held.inputs.sigma)
    cond_pred=post.predict(cond.inputs,sigma=cond.inputs.sigma)
    samples=post.sample(config.posterior_samples,generator=torch.Generator().manual_seed(seed))
    probe=slice_query(next(iter(queries.values())),0,37)
    full=samples.evaluate(probe);chunks=torch.cat([samples.evaluate(slice_query(probe,i,i+7)) for i in range(0,37,7)],1)
    centered=samples.xi-samples.xi.mean(0);emp=centered.T@centered/(config.posterior_samples-1)
    scores={}
    for label,obs in (('conditioning_replicate',cond),('heldout_independent',held)):
        score=post.log_predictive_density(obs.inputs); scores[label]={k:v for k,v in score.items() if k!='marginal'}
        scores[label]['marginal_per_valid_entry']=score['marginal'][obs.inputs.valid].flatten().tolist()
    truth=case.tensor()
    H,y,sigma=observed_system(session.spec,cond.inputs)
    independent=condition_covariance(H,y,sigma,post.prior.mean,post.prior.covariance)
    return {'coefficient_posterior_mean':post.mean.tolist(),'coefficient_posterior_covariance':post.covariance.tolist(),
            'matrix_diagnostics':post.diagnostics,'field_quantities':fields,
            'independent_covariance_form_max_discrepancy':{'mean':float((post.mean-independent.mean).abs().max()),'covariance':float((post.covariance-independent.covariance).abs().max())},
            'coefficient_coverage':coverage(post.mean,post.covariance.diag(),truth),
            'heldout_latent_coverage':coverage(held_pred.mean,held_pred.latent_variance,held.metadata.clean,held.inputs.valid),
            'heldout_measurement_coverage':coverage(held_pred.mean,held_pred.variance,held.inputs.value,held.inputs.valid),
            'heldout_variance':{'latent':held_pred.latent_variance.flatten().tolist(),'measurement':held_pred.variance.flatten().tolist(),
                                'sigma_squared':held.inputs.sigma.square().flatten().tolist(),
                                'variance_addition_max_error':float((held_pred.variance-held_pred.latent_variance-held.inputs.sigma.square()).abs().max())},
            'conditioning_replicate_measurement_coverage':coverage(cond_pred.mean,cond_pred.variance,cond.inputs.value,cond.inputs.valid),
            'log_predictive_density':scores,
            'coherent_samples':{'count':config.posterior_samples,'seed':seed,'chunk_max_discrepancy':float((full-chunks).abs().max()),
                                'empirical_mean_max_error':float((samples.xi.mean(0)-post.mean).abs().max()),
                                'empirical_covariance_max_error':float((emp-post.covariance).abs().max()),
                                'interpretation':'finite Monte Carlo check; one coefficient vector per entire field, not posterior mean'}}, arrays


def evaluate(plan, methods, normalizers, *, config=EvaluationConfig(), output=None, allow_final_test=False):
    """One fixed context per case/method; cold preparation measured after warmup.

    Deterministic scientific records and noisy performance measurements are
    separate. References never enter Method.prepare or Session.predict.
    """
    plan.validate(allow_final_test=allow_final_test)
    require(plan.config_identity==config.identity and normalizers.identity==plan.normalizer_identity,'plan/config/normalizer mismatch')
    require(len({m.name for m in methods})==len(methods) and methods,'distinct method names required')
    models={id(m.model):m.model for m in methods if m.model is not None}
    fingerprints={k:model_identity(m) for k,m in models.items()};modes={k:[(x,x.training) for x in m.modules()] for k,m in models.items()}
    for m in models.values():m.eval()
    rows={m.name:[] for m in methods};costs={m.name:[] for m in methods};assets={}; generation=[]
    start=time.perf_counter()
    try:
        with torch.inference_mode(False):
            for entry in plan.cases:
                spec=canonical_spec(entry.pde);t=time.perf_counter();case,cond,held=plan.regenerate(entry,normalizers)
                data_seconds=time.perf_counter()-t;observed_id=input_identity(cond.inputs)
                queries={('v' if q else 'u'):plan.query(entry.pde,q) for q in dict(plan.quantities)[entry.pde]}
                t=time.perf_counter();targets={q:evaluate_batch((spec,),case_coefficients((case,)),query) for q,query in queries.items()}
                reference_seconds=time.perf_counter()-t
                generation.append({'physical_id':entry.physical_id,'dataset_including_sensor_reference_seconds':data_seconds,'full_grid_reference_seconds':reference_seconds})
                # This exact posterior is an explicitly timed scoring comparator,
                # even when D is absent from the selected output methods.
                t=time.perf_counter();dpost=ExactGaussianD().fit(spec,cond.inputs);d_reference_seconds=time.perf_counter()-t
                generation[-1]['D_scoring_comparator_posterior_seconds_shared_once_per_case']=d_reference_seconds
                probe=diagnostic_query(plan,spec)
                for method in methods:
                    if config.warmup:
                        warm=method.prepare(spec,cond.inputs);warm.predict(slice_query(probe,0,4),chunk=4);sync(method.device);del warm
                    sync(method.device);t=time.perf_counter();session=method.prepare(spec,cond.inputs);sync(method.device);prepare_seconds=time.perf_counter()-t
                    sync(method.device);t=time.perf_counter();preds={q:session.predict(query,chunk=config.chunk) for q,query in queries.items()};sync(method.device);query_seconds=time.perf_counter()-t
                    # Same prepared context for chunks, individual/batched probes and all scoring queries.
                    p_all=session.predict(probe,chunk=len(plan.diagnostic_points));p_chunks=session.predict(probe,chunk=7)
                    p_individual=torch.cat([session.predict(slice_query(probe,i,i+1),chunk=1) for i in range(probe.valid.shape[1])],1)
                    row={'architecture':method.architecture,'training_status':method.status,'checkpoint':method.checkpoint,
                         'pde':entry.pde,'case_id':entry.physical_id,'split':plan.split,'track':'T1','case_identity':case.identity,
                         'conditioning_identity':cond.metadata.identity,'heldout_identity':held.metadata.identity,'raw_conditioning_input_identity':observed_id,
                         'neural_input_precision':'explicit float32 copy of same raw records' if method.model is not None else 'CPU float64',
                         'quantities':{},'time_errors':{},'coefficient_vs_hidden_truth':coefficient_errors(session.xi,case.tensor(),prior_scales(entry.pde)),
                         'coefficient_vs_D_posterior_mean':coefficient_errors(session.xi,dpost.mean,prior_scales(entry.pde)),
                         'chunk_max_discrepancy':float((p_all-p_chunks).abs().max()),'individual_max_discrepancy':float((p_all-p_individual).abs().max())}
                    asset={'position':queries['u'].position.numpy(),'physical_time':queries['u'].physical_time.numpy(),
                           'conditioning_position':cond.inputs.position.numpy(),'conditioning_time':cond.inputs.physical_time.numpy(),
                           'conditioning_quantity':cond.inputs.quantity_id.numpy(),'conditioning_y':cond.inputs.value.numpy(),
                           'heldout_position':held.inputs.position.numpy(),'heldout_time':held.inputs.physical_time.numpy(),
                           'heldout_quantity':held.inputs.quantity_id.numpy(),'heldout_y':held.inputs.value.numpy(),
                           'conditioning_prediction':session.predict(cond.inputs,chunk=config.chunk).numpy(),
                           'heldout_prediction':session.predict(held.inputs,chunk=config.chunk).numpy()}
                    for quantity,q in queries.items():
                        scale=normalizers.scale(entry.pde,1 if quantity=='v' else 0)
                        row['quantities'][quantity]=errors(preds[quantity],targets[quantity],scale,q.valid)
                        row['time_errors'][quantity]=[]
                        if entry.pde!='poisson':
                            for ti in plan.dynamic_times:
                                valid=q.valid & (q.physical_time[...,0]==ti)
                                row['time_errors'][quantity].append({'physical_time':ti,**errors(preds[quantity],targets[quantity],scale,valid)})
                        asset['truth_'+quantity]=targets[quantity].numpy();asset['prediction_'+quantity]=preds[quantity].numpy()
                    row['conditioning_sensors']=sensor_scores(session,cond,normalizers,config.chunk)
                    row['heldout_sensors']=sensor_scores(session,held,normalizers,config.chunk)
                    t=time.perf_counter()
                    if config.physical_diagnostics:row['physics']=physical_diagnostics(session,plan,case.tensor(),normalizers)
                    if method.architecture.startswith('B') or method.architecture=='Fixed':row['refinement']=refinement_scores(session,case.tensor(),dpost.mean,cond.inputs)
                    if method.architecture=='D':
                        row['uncertainty'],variance=d_uncertainty(session,case,cond,held,queries,targets,config,plan.evaluation_seed+entry.case_id)
                        for quantity,value in variance.items():asset['latent_variance_'+quantity]=value
                    else:row['uncertainty']={'available':False,'reason':'deterministic estimate; no justified posterior'}
                    if config.observation_diagnostics:row['observation_diagnostics']=observation_diagnostics(method,spec,cond.inputs,probe,baseline=session)
                    diagnostic_seconds=time.perf_counter()-t
                    require(input_identity(cond.inputs)==observed_id,'inference/diagnostics mutated raw observations')
                    nqueries=sum(int(q.valid.sum()) for q in queries.values()); n=cond.inputs.valid.shape[1]
                    iterations=len(getattr(session.context,'trajectory',torch.empty(1,1,1))[0])-1 if method.architecture.startswith('B') or method.architecture=='Fixed' else 0
                    physical_cost={'refinement_iterations':iterations,'measurement_forward_evaluations':4 if method.architecture.startswith('B') else 6 if method.architecture=='Fixed' else 0,
                                   'scalar_measurement_evaluations':(4 if method.architecture.startswith('B') else 6 if method.architecture=='Fixed' else 0)*n,
                                   'B1_anchor_projection_fields':2*iterations if method.architecture=='B1' and method.ablation.spatial else 0,
                                   'D_posterior_factorizations':2 if method.architecture=='D' else 0,
                                   'modal_query_field_chunks':sum(math.ceil(q.valid.shape[1]/config.chunk) for q in queries.values()) if method.architecture in {'B0','B1','D','Fixed','Prior'} else 0,
                                   'B_internal_modal_quantity_evaluations':sum(math.ceil(q.valid.shape[1]/config.chunk) for q in queries.values())*(2 if entry.pde=='wave' else 1) if method.architecture.startswith('B') else 0,
                                   'D_latent_variance_batches':sum(math.ceil(q.valid.shape[1]/config.chunk) for q in queries.values()) if method.architecture=='D' else 0,
                                   'coordinate_query_scalar_values':nqueries,'iterative_mesh_solves':0}
                    costs[method.name].append({'case_id':entry.physical_id,'conditioning_refinement_or_posterior_seconds':prepare_seconds,
                                              'full_grid_queries_seconds':query_seconds,'inference_total_seconds':prepare_seconds+query_seconds,
                                              'scoring_diagnostics_seconds':diagnostic_seconds,
                                              'warmup':config.warmup,'query_chunk':config.chunk,'case_batch':1,'physical_work':physical_cost,'memory':memory_report(method.device.type)})
                    rows[method.name].append(row);assets[method.name+'__'+entry.physical_id.replace('/','_')]=asset
    finally:
        for k,model in models.items():
            for module,mode in modes[k]:module.training=mode
    require(all(model_identity(m)==fingerprints[k] for k,m in models.items()),'evaluation mutated model weights/buffers')
    aggregate_records={name:aggregate(cases) for name,cases in rows.items()}
    paired={name:paired_interval(cases,rows['D'],seed=plan.evaluation_seed) for name,cases in rows.items() if 'D' in rows and name!='D'}
    cov={}
    if 'D' in rows:
        for key in ('coefficient_coverage','heldout_latent_coverage','heldout_measurement_coverage'):
            cov[key]=coverage_interval([{'pde':r['pde'],'coverage':r['uncertainty'][key]} for r in rows['D']])
        cov['field_latent_coverage']=coverage_interval([{'pde':r['pde'],'coverage':{k:sum(v['latent_coverage'][k] for v in r['uncertainty']['field_quantities'].values())/len(r['quantities']) for k in ('0.5','0.9','0.95')}} for r in rows['D']])
    science={'version':'matched-results-v1','plan_identity':plan.identity,'source_identity':plan.source_identity,'config':asdict(config),
             'methods':{m.name:{**{k:v for k,v in m.metadata().items() if k!='training'},'training_selection':{k:m.training.get(k) for k in ('successful_updates','declared_updates','effective_batch','cases_processed','root_seed','device','source_identity','config_identity')}} for m in methods},'cases':rows,'aggregate':aggregate_records,
             'paired_standardized_MSE_minus_D':paired,'D_coverage_intervals':cov,
             'interpretation':'D posterior mean is Bayes-optimal in expected squared error under correctly specified T1; tiny smoke comparisons are descriptive, not rankings',
             'apparent_advantage_checks':{name:('inspect leakage, precision/metric consistency and statistical variation before claiming superiority' if v['estimate']<0 else 'no aggregate advantage observed') for name,v in paired.items()},
             'inference_interfaces':'known spec + measured ObservationBatch only; reference tensors score-only',
             'comparison_protocol_evidence':{'successful_updates':{m.name:m.training.get('successful_updates') for m in methods if m.model is not None},
                   'wall_clock_matched':False,'independent_neural_seed_count':len({m.training.get('root_seed') for m in methods if m.training}),
                   'serious_comparison':False,'case_selection':'fixed IDs in plan; not chosen from errors'},
             'weights_unchanged':True,'test_evaluated':plan.split=='test','optimizer_steps':0,'geometry':'canonical unit square only; no irregular-domain evidence'}
    identity=digest(science)
    performance={'elapsed_seconds':time.perf_counter()-start,'dataset_reference_costs':generation,'per_case':costs,
                 'training_metadata':{m.name:m.training for m in methods},'device':{m.name:str(m.device) for m in methods},
                 'summary':{name:{'median_seconds':float(np.median([r['inference_total_seconds'] for r in values])),
                                   'p95_seconds':float(np.quantile([r['inference_total_seconds'] for r in values],.95)) if len(values)>=20 else None,
                                   'p95_note':'requires >=20 physical cases; tiny mixed-PDE median descriptive only','cases':len(values)} for name,values in costs.items()}}
    result={'identity':identity,'scientific':science,'performance':performance}
    if output is not None:
        directory=Path(output)/plan.identity;directory.mkdir(parents=True,exist_ok=True);plan.save(directory)
        persist_json(directory/f'{identity}.json',science)
        (directory/'performance.json').write_text(json.dumps(performance,indent=2)+'\n')
        for key,asset in assets.items():np.savez_compressed(directory/f'{key}.npz',**asset)
        result['directory']=str(directory.resolve())
        (directory/'result.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

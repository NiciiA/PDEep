"""Fixed held-out validation without optimizer backward or training-state mutation."""
from dataclasses import asdict
from collections import defaultdict
import math,time
import torch
from .train_types import ValidationReport
from .b_physics import physical_batch,cpu64
from .core import require,digest


def model_identity(model):
    from .b_artifacts import tree_identity,tree_cpu
    return digest(tree_identity(tree_cpu(model.state_dict())))


def _case_metrics(spec,prediction,xi,active,batch,normalizers,sensor_prediction):
    pred=prediction.detach().to('cpu').to(torch.float64);truth=cpu64(batch.target.detach());q=batch.queries
    quantities={}
    for quantity in (0,1):
        mask=(q.valid[0]&(q.quantity_id[0]==quantity)).to('cpu')
        if bool(mask.any()):
            e=(pred-truth)[0,mask,0];scale=normalizers.scale(spec.name,quantity)
            sm=(batch.observations.valid[0]&(batch.observations.quantity_id[0]==quantity)).to('cpu')
            r=(cpu64(sensor_prediction.detach())-cpu64(batch.observations.masked().value))[0,sm,0];sigma=cpu64(batch.observations.sigma.detach())[0,sm,0]
            quantities[str(quantity)]={'physical_MSE':float(e.square().mean()),'normalized_MSE':float((e/scale).square().mean()),'queries':int(mask.sum()),
                                      'sensor_physical_MSE':float(r.square().mean()) if r.numel() else None,
                                      'sensor_noise_standardized_MSE':float((r/sigma).square().mean()) if r.numel() else None,
                                      'sensor_Gaussian_NLL_sum':float((.5*((r/sigma).square()+2*sigma.log()+math.log(2*math.pi))).sum()),'sensors':int(sm.sum())}
    a=active.to('cpu');target=cpu64(batch.unknowns.masked());xi=cpu64(xi.detach())
    scales=torch.tensor([v for block in spec.parameter_blocks for v in block.prior.scale],dtype=torch.float64)
    error=(xi-target)[0,a[0]]
    return {'pde':spec.name,'quantities':quantities,'coefficient_normalized_MSE':float((error/scales).square().mean()),'coefficient_physical_MSE':float(error.square().mean())}


def _aggregate(cases):
    grouped=defaultdict(list)
    for case in cases:grouped[case['pde']].append(case)
    result={}
    for pde,rows in grouped.items():
        quantities={}
        for q in sorted({q for row in rows for q in row['quantities']}):
            items=[row['quantities'][q] for row in rows if q in row['quantities']]
            # Equal case weighting, not pooled-point weighting; query counts recorded.
            metrics={key:sum(v[key] for v in items if v[key] is not None)/max(sum(v[key] is not None for v in items),1)
                     for key in ('physical_MSE','normalized_MSE','sensor_physical_MSE','sensor_noise_standardized_MSE','sensor_Gaussian_NLL_sum')}
            metrics['physical_RMSE']=math.sqrt(metrics['physical_MSE']);metrics['queries']=sum(v['queries'] for v in items);metrics['sensors']=sum(v['sensors'] for v in items);metrics['cases']=len(items)
            quantities[q]=metrics
        result[pde]={'cases':len(rows),'quantities':quantities,'field_normalized_MSE':sum(sum(v['normalized_MSE'] for v in row['quantities'].values())/len(row['quantities']) for row in rows)/len(rows),
                     'coefficient_normalized_MSE':sum(v['coefficient_normalized_MSE'] for v in rows)/len(rows),
                     'coefficient_physical_MSE':sum(v['coefficient_physical_MSE'] for v in rows)/len(rows)}
    result['common']={'cases':len(cases),'field_normalized_MSE':sum(sum(v['normalized_MSE'] for v in c['quantities'].values())/len(c['quantities']) for c in cases)/len(cases),
                      'coefficient_normalized_MSE':sum(c['coefficient_normalized_MSE'] for c in cases)/len(cases)}
    return result


def validate(model,architecture,manifest,normalizers,*,device='cpu',successful_update=0,checkpoint_identity=None):
    """No .backward, optimizer call, RNG draws or parameter-gradient zeroing.

    C physical residuals explicitly re-enable grad even under an outer no_grad/
    inference_mode. Context/field-only passes use no_grad. Original module modes
    are restored individually in finally, including mixed train/eval submodules.
    """
    start=time.perf_counter();modes=[(m,m.training) for m in model.modules()];cases=[];specific=[]
    checkpoint_identity=model_identity(model) if checkpoint_identity is None else checkpoint_identity
    try:
        model.eval()
        with torch.inference_mode(False):
            for batch in manifest.batches(normalizers,device=device):
                spec=batch.specs[0]
                with torch.inference_mode(False),torch.no_grad():
                    if architecture in {'B0','B1'}:
                        result=model.infer_unknowns(model.prepare_context(batch.specs,batch.observations))
                        pred=model.predict_queries(result,batch.queries).predicted_value;sensor=model.predict_queries(result,batch.observations).predicted_value
                        xi,active=result.xi_hat,result.xi_active
                        specific.append({'pde':spec.name,'Phi_by_iteration':result.objectives[0].tolist()})
                    else:
                        output=model(batch.specs,batch.observations,batch.queries);context=output.context
                        pred=output.predicted_value;xi=output.xi_hat;active=output.xi_active;sensor=model.predict_queries(context,batch.observations).predicted_value
                        # Existing exact objectives give architecture-specific diagnostics.
                        from .a_training import objective as base_objective
                        loss=base_objective(output,batch)
                        specific.append({'pde':spec.name,'field_objective':float(loss['field']),'xi_objective':float(loss['xi'])})
                if architecture=='C':
                    from .constraints import physical_losses,strong_diagnostics
                    from .c_checks import independent_diagnostic_queries
                    # Regenerate context with gradient tracking, not inference tensors.
                    # Only coordinate gradients are requested; .grad on parameters is untouched.
                    with torch.inference_mode(False),torch.enable_grad():
                        context=model.prepare_context(batch.specs,batch.observations)
                        physical=physical_losses(model,context,create_graph=False)
                        strong=strong_diagnostics(model,context,independent_diagnostic_queries(batch))
                        specific[-1].update(weak_objective=float(physical.weak_per_case.detach().mean()),initial_objective=float(physical.initial_per_case.detach().mean()),
                                            strong_physical_RMSE=strong[0]['physical_RMSE'],strong_normalized_RMSE=strong[0]['normalized_RMSE'])
                cases.append(_case_metrics(spec,pred,xi,active,batch,normalizers,sensor))
            metrics=_aggregate(cases)
            return ValidationReport(architecture,checkpoint_identity,manifest.identity,manifest.name,successful_update,metrics,{'per_case':specific,'training_objectives_are_not_cross_architecture_rankings':True},
                                    {'cases':len(cases),'queries':sum(v['queries'] for c in cases for v in c['quantities'].values()),'case_ids':list(manifest.case_ids)},time.perf_counter()-start,device,'float32; B physics explicitly CPU64' if architecture.startswith('B') else 'float32')
    finally:
        for module,mode in modes:module.training=mode


def validate_D(manifest,normalizers):
    """Exact D evaluation-only; posterior uncertainty, no optimizer interface."""
    from .bayes import ExactGaussianD
    start=time.perf_counter();cases=[];uncertainty=[]
    for batch in manifest.batches(normalizers,device='cpu'):
        spec=batch.specs[0];post=ExactGaussianD().fit(spec,physical_batch(batch.observations))
        pred=post.predict(physical_batch(batch.queries));sensor=post.predict(physical_batch(batch.observations)).mean
        xi=torch.nn.functional.pad(post.mean,(0,18-len(post.mean)))[None]
        cases.append(_case_metrics(spec,pred.mean,xi,batch.unknowns.active,batch,normalizers,sensor))
        uncertainty.append({'pde':spec.name,'pointwise_latent_variance_mean':float(pred.variance.mean()),'covariance_trace':float(post.covariance.trace())})
    return ValidationReport('D','exact-T1-Gaussian',manifest.identity,manifest.name,0,_aggregate(cases),{'posterior_uncertainty':uncertainty},
                            {'cases':len(cases),'queries':sum(v['queries'] for c in cases for v in c['quantities'].values())},time.perf_counter()-start,'cpu','float64')

"""Rich diagnostic bundles are inspectable evidence, never resumable checkpoints."""
from dataclasses import fields
import math
import torch
from .artifact_io import write_artifact
from .artifact_metadata import make_metadata
from .artifact_tensors import tree_cpu
from .inference_artifacts import pack_batch
from .core import require


def summary(value):
    if isinstance(value,torch.Tensor):
        t=value.detach();finite=torch.isfinite(t);v=t[finite]
        return {'shape':list(t.shape),'dtype':str(t.dtype),'device':str(t.device),'nonfinite':int((~finite).sum()),
                'min':float(v.min()) if v.numel() else None,'max':float(v.max()) if v.numel() else None}
    if isinstance(value,dict):return {str(k):summary(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [summary(v) for v in value]
    return str(value) if isinstance(value,float) and not math.isfinite(value) else value


def safe_scalars(value):
    if isinstance(value,dict):return {k:safe_scalars(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(safe_scalars(v) for v in value)
    return str(value) if isinstance(value,float) and not math.isfinite(value) else value


def capture_intermediates(architecture,model,spec,observations,queries):
    """All requested modules/intermediate states; no fabricated attention maps."""
    from .representations import float32_batch
    if architecture=='D':
        from .bayes import ExactGaussianD
        post=ExactGaussianD().fit(spec,observations);prediction=post.predict(queries)
        return {'mean':post.mean,'covariance':post.covariance,'matrix_diagnostics':post.diagnostics,
                'posterior_mean_field':prediction.mean,'latent_variance':prediction.latent_variance}
    device=next(model.parameters()).device;device=device.type if device.type=='mps' else str(device)
    obs=float32_batch(observations,device=device);q=float32_batch(queries,device=device)
    modes=[(m,m.training) for m in model.modules()]
    try:
        model.eval()
        with torch.no_grad():
            if architecture.startswith('B'):
                result=model.infer_unknowns(model.prepare_context((spec,),obs),return_diagnostics=True)
                return tree_cpu({'trajectory':result.trajectory,'objectives':result.objectives,'sensor_predictions':result.predictions,
                                 'residuals':result.residuals,'gradients':result.gradients,'preconditioner':result.diagonal,
                                 'steps':result.steps,'coefficient_memory':result.token_history,'spatial_memory':result.spatial_history,
                                 'equation_nodes':result.context.graph.nodes,'observation_tokens':result.context.observations.tokens,
                                 'predicted_field':model.predict_queries(result,queries).physical_value})
            context=model.prepare_context((spec,),obs,return_attention=True)
            result={'equation_nodes':context.graph.nodes,'equation_context':context.graph.context,
                    'observation_tokens':context.observations.tokens,'observation_summary':context.observations.summary,
                    'xi':model.infer_unknowns(context).xi_hat,'prediction':model.predict_queries(context,q).predicted_value}
            if architecture=='A':result.update(memory=context.memory.tokens,attention=context.memory.attention)
            else:result.update(interaction_tokens=context.interaction_tokens,attention=context.attention,conditioning=context.conditioning)
            return tree_cpu(result)
    finally:
        for module,mode in modes:module.training=mode


def save_diagnostic_bundle(directory,*,architecture,normalizers,run_id,model=None,config=None,successful=0,
                           packet=None,inputs=None,intermediates=None,gradients=None,parents=(),failed=False):
    metadata=make_metadata('diagnostic_bundle',architecture,normalizers,model=model,config=config,run_id=run_id,
                           successful=successful,parents=parents)
    content=safe_scalars({'packet':packet or {},'inputs':inputs or {},'intermediates':intermediates or {},'gradients':gradients or {}})
    metadata.update(diagnostic_summary=summary(content),safe_to_resume=False,
                    diagnostic_status='failure evidence; may intentionally contain nonfinite tensors' if failed else 'bounded diagnostic evidence')
    payload={'version':'diagnostic-v2','architecture':architecture,'model':{} if model is None else tree_cpu(model.state_dict()),**tree_cpu(content)}
    return write_artifact(directory,None,metadata,payload,required=('model','inputs','intermediates','packet','gradients'),idempotent=True)


def save_failure_bundle(engine,packet,batch):
    inputs={}
    if batch is not None:
        inputs={'problem_identities':[s.identity for s in batch.specs],'observations':pack_batch(batch.observations),
                'queries':pack_batch(batch.queries),'target':batch.target,'unknowns':batch.unknowns.xi,'active':batch.unknowns.active}
    gradients={name:p.grad for name,p in engine.model.named_parameters() if p.grad is not None}
    parents=[] if engine.state.last_checkpoint is None else [{'path_hint':engine.state.last_checkpoint,'role':'last completed successful boundary; this failure bundle cannot resume'}]
    return save_diagnostic_bundle(engine.run_dir/'diagnostics',architecture=engine.config.architecture,
                                  normalizers=engine.normalizers,run_id=engine.config.run_id,model=engine.model,
                                  config=engine.config,successful=engine.state.successful,packet=packet,inputs=inputs,
                                  gradients=gradients,parents=parents,failed=True)

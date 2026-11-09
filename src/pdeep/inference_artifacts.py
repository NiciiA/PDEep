"""Complete portable inference exports, independent of any training workflow.

No optimizer, trainer or training-objective module is imported on reload.
All encoders, memories, heads, vocabulary, normalization and physical dependencies
are validated. D persists measured inputs/prior/spec, and recomputes its posterior.
"""
from dataclasses import dataclass,fields
from pathlib import Path
import torch
from .core import require,digest
from .batches import ObservationBatch,QueryBatch
from .data import CaseDataset,observations,training_queries
from .reference import canonical_spec
from .representations import float32_batch
from .artifact_io import write_artifact,read_payload
from .artifact_metadata import make_metadata,validate_metadata
from .artifact_tensors import tree_cpu,state_identity
from .runtime import resolve_device


def pack_batch(batch):return {f.name:tree_cpu(getattr(batch,f.name)) for f in fields(batch)}
def unpack_observation(value):return ObservationBatch(**value)
def unpack_query(value):return QueryBatch(**value)


def inference_examples(normalizers, *, count=8):
    ds=CaseDataset(normalizers.root_seed);result=[]
    for pde in ('poisson','heat','wave'):
        case=ds.case('validation',pde,0)
        obs=observations(ds,case,normalizers,set_id='artifact-v2-independent-example')
        q=training_queries(ds,case,count=count,set_id='artifact-v2-independent-query')
        result.append({'pde':pde,'observation_identity':obs.metadata.identity,'query_identity':q.identity,
                       'observations':pack_batch(obs.inputs),'queries':pack_batch(q.inputs)})
    return result

@dataclass
class InferenceContext:
    owner: object
    spec: object
    prepared: object
    xi: torch.Tensor

@dataclass
class InferenceModel:
    architecture: str
    model: object
    metadata: dict
    payload: dict
    device: str='cpu'

    def prepare(self,spec,observations):
        require(spec==canonical_spec(spec.name),'export supports canonical T1 problem specifications only')
        observations.validate((spec,))
        with torch.no_grad():
            if self.architecture=='D':
                from .bayes import ExactGaussianD
                state=ExactGaussianD().fit(spec,observations);xi=state.mean[None]
            else:
                obs=float32_batch(observations,device=self.device)
                state=self.model.prepare_context((spec,),obs)
                if self.architecture.startswith('B'):
                    state=self.model.infer_unknowns(state);xi=state.xi_hat
                else:xi=self.model.infer_unknowns(state).xi_hat
        return InferenceContext(self,spec,state,tree_cpu(xi).double())

    def predict(self,context,queries,*,chunk=1024):
        require(context.owner is self,'inference context belongs to another loaded model')
        require(type(chunk) is int and chunk>0,'positive query chunk required');queries.validate((context.spec,));values=[]
        with torch.no_grad():
            for lo in range(0,queries.valid.shape[1],chunk):
                q=QueryBatch(*(getattr(queries,f.name)[:,lo:lo+chunk] for f in fields(QueryBatch)))
                if self.architecture=='D':value=context.prepared.predict(q).mean
                elif self.architecture.startswith('B'):value=self.model.predict_queries(context.prepared,q).physical_value
                else:value=self.model.predict_queries(context.prepared,float32_batch(q,device=self.device)).predicted_value
                values.append(value.detach().cpu().double())
        return torch.cat(values,1) if values else torch.zeros(1,0,1,dtype=torch.float64)


def export_inference(model,directory,*,architecture,normalizers,run_id,config=None,successful=0,
                     examples=None,parents=(),status='untrained'):
    require(architecture in {'A','C','B0','B1','D'},'exact architecture required')
    require((model is None)==(architecture=='D'),'D has no neural state; neural exports need a complete model')
    examples=inference_examples(normalizers) if examples is None else examples
    metadata=make_metadata('inference_export',architecture,normalizers,model=model,run_id=run_id,config=config,
                           successful=successful,parents=parents)
    metadata['training_status']=status
    metadata['example_tolerance']={'rtol':2e-11,'atol':2e-12} if architecture=='D' else {'rtol':2e-6,'atol':2e-7}
    modes=[] if model is None else [(m,m.training) for m in model.modules()]
    if model is not None:model.eval()
    device='cpu' if model is None else next(model.parameters()).device.type
    if device=='cuda':device=str(next(model.parameters()).device)
    wrapper=InferenceModel(architecture,model,metadata,{},device)
    expected=[]
    try:
        for example in examples:
            spec=canonical_spec(example['pde']);obs=unpack_observation(example['observations']);q=unpack_query(example['queries'])
            ctx=wrapper.prepare(spec,obs);value=wrapper.predict(ctx,q,chunk=3)
            record={**example,'expected_prediction':value,'expected_xi':ctx.xi}
            if architecture=='D':record['expected_covariance']=ctx.prepared.covariance.clone()
            expected.append(record)
    finally:
        for module,mode in modes:module.training=mode
    payload={'version':'inference-v2','architecture':architecture,'model':{} if model is None else tree_cpu(model.state_dict()),'examples':expected}
    return write_artifact(directory,None,metadata,payload,required=('model','examples','version','architecture'),idempotent=True)


def load_inference(manifest,*,expected_architecture,device='cpu',verify_examples=True):
    record,payload=read_payload(manifest,kind='inference_export',architecture=expected_architecture,required=('model','examples','version','architecture'))
    require(payload['version']=='inference-v2' and payload['architecture']==expected_architecture,'inference payload version/architecture mismatch')
    norm,vocab=validate_metadata(record);arch=expected_architecture
    if arch=='D':
        resolve_device(device,'float64',reference=True);require(not payload['model'],'D export must not contain neural weights');model=None
    else:
        resolved,_=resolve_device(device,'float32')
        from .models import select_architecture
        seed=record['training_config']['root_seed'] if record['training_config'] is not None else norm.root_seed
        with torch.random.fork_rng(devices=[]):model=select_architecture(arch,normalizers=norm,vocabulary=vocab,root_seed=seed)
        validate_metadata(record,model=model)
        model.load_state_dict(payload['model'],strict=True)
        require(all(bool(torch.isfinite(v).all()) for v in model.state_dict().values() if v.is_floating_point()),'nonfinite inference weights')
        require(state_identity(model)==record['component_inventory']['model_state_identity'],'model tensor inventory/identity mismatch')
        model.to(resolved).eval()
    wrapper=InferenceModel(arch,model,record,payload,device)
    if verify_examples:verify_inference_examples(wrapper)
    return wrapper


def verify_inference_examples(wrapper):
    checks=[];tol=wrapper.metadata['example_tolerance']
    require(wrapper.payload['examples'],'reproducible inference examples missing')
    for example in wrapper.payload['examples']:
        ctx=wrapper.prepare(canonical_spec(example['pde']),unpack_observation(example['observations']))
        pred=wrapper.predict(ctx,unpack_query(example['queries']),chunk=3);expected=example['expected_prediction']
        error=float((pred-expected).abs().max());xi_error=float((ctx.xi-example['expected_xi']).abs().max())
        require(torch.allclose(pred,expected,**tol),'independent inference example predictions differ beyond declared tolerance')
        require(torch.allclose(ctx.xi,example['expected_xi'],**tol),'independent inference coefficient predictions differ')
        row={'pde':example['pde'],'prediction_max_error':error,'coefficient_max_error':xi_error,'tolerance':tol}
        if wrapper.architecture=='D':
            row['covariance_max_error']=float((ctx.prepared.covariance-example['expected_covariance']).abs().max())
            require(torch.allclose(ctx.prepared.covariance,example['expected_covariance'],**tol),'D independently reconstructed covariance differs')
        checks.append(row)
    return {'passed':True,'architecture':wrapper.architecture,'device':wrapper.device,'cases':checks}

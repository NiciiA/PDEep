"""Explicit architecture capabilities and adapters to unchanged scientific losses."""
from dataclasses import dataclass
import torch
from .core import require
from .train_types import LossReport


@dataclass(frozen=True)
class ArchitectureRegistration:
    identity: str
    trainable: bool
    pdes: tuple=('poisson','heat','wave')
    tracks: tuple=('T1',)
    inputs: tuple=('ProblemSpec','ObservationBatch','QueryBatch')
    outputs: tuple=('field','physical_coefficients')
    physical_gradients: bool=False
    coordinate_derivatives: bool=False
    devices: tuple=('cpu','mps','cuda')
    precisions: tuple=('float32',)
    checkpoint_version: str='pdeep-artifact-v2/training_checkpoint'
    objective: str=''
    objective_implementation: str=''
    device_policy: str='explicit neural device; reference targets CPU64'


REGISTRY={
 'A':ArchitectureRegistration('A',True,objective='L_field + 0.1 L_xi',objective_implementation='pdeep.a_training.objective'),
 'C':ArchitectureRegistration('C',True,coordinate_derivatives=True,objective='L_field + 0.1 L_xi + 0.1 L_weak + 0.1 L_initial',objective_implementation='pdeep.c_training.objective'),
 'B0':ArchitectureRegistration('B0',True,physical_gradients=True,objective='L_xi(final)',objective_implementation='pdeep.b_training.coefficient_loss',device_policy='neural selected-device32; explicit differentiable CPU64 physics bridge'),
 'B1':ArchitectureRegistration('B1',True,physical_gradients=True,objective='L_xi(final)',objective_implementation='pdeep.b_training.coefficient_loss',device_policy='neural selected-device32; explicit differentiable CPU64 physics bridge'),
 'D':ArchitectureRegistration('D',False,inputs=('ProblemSpec','ObservationBatch'),outputs=('posterior_mean','posterior_covariance','predictive_variance'),devices=('cpu',),precisions=('float64',),checkpoint_version='exact-Gaussian-D-v1',objective='none; exact Gaussian conditioning')}


def registration(name):
    require(name in REGISTRY,'select exact A/C/B0/B1/D identity; B is ambiguous')
    return REGISTRY[name]


class ScientificAdapter:
    """No model copies, toy paths, changed weights or extra objectives."""
    def __init__(self,architecture):
        self.registration=registration(architecture)
        require(self.registration.trainable,'D has no neural training objective')
        self.architecture=architecture

    def loss(self,model,batch,*,detailed=False):
        counts={'cases':len(batch.specs),'valid_queries':int(batch.queries.valid.sum()),'active_coefficients':int(batch.unknowns.active.sum()),'valid_observations':int(batch.observations.valid.sum())}
        if self.architecture in {'B0','B1'}:
            from .b_training import coefficient_loss
            result=model.infer_unknowns(model.prepare_context(batch.specs,batch.observations),return_diagnostics=detailed)
            value=coefficient_loss(result,batch.unknowns)
            diagnostics={'Phi_mean_by_iteration':result.objectives.detach().mean(0).tolist(),'physical_device':'cpu','physical_dtype':'float64','token_shape':list(result.token_history[0].shape) if detailed else None}
            if detailed and result.spatial_history:
                diagnostics['spatial']=[{'tokens':tensor_summary(s['tokens']),'coarse':tensor_summary(s['coarse_summary']),'fine':tensor_summary(s['fine_summary']),
                                         'attention':[tensor_summary(w) for w in s['attention']]} for s in result.spatial_history]
            return LossReport(value,{'xi':value},{'xi':value},{'xi':'dimensionless mean-case active prior-standardized coefficient MSE'},counts,diagnostics)
        output=model(batch.specs,batch.observations,batch.queries,return_attention=detailed)
        if self.architecture=='A':
            from .a_training import objective
            losses=objective(output,batch,coefficient_weight=.1)
            raw={k:losses[k] for k in ('field','xi')};weighted={'field':raw['field'],'xi':.1*raw['xi']};diagnostics={}
        else:
            from .c_training import objective
            losses=objective(model,output,batch,coefficient_weight=.1,weak_weight=.1,initial_weight=.1)
            raw={k:losses[k] for k in ('field','xi','weak','initial')};weighted={k:losses['weighted_'+k] for k in raw}
            diagnostics={'quadrature':losses['physical'].timing,'spatial_order':16,'time_order':8,'initial_cases':sum(s.name!='poisson' for s in batch.specs)}
        meanings={'field':'dimensionless query MSE: mean valid queries/quantity, available quantities/case, cases',
                  'xi':'dimensionless mean-case active prior-standardized coefficient MSE',
                  'weak':'dimensionless family-specific prior-term normalized weak integral MSE, temporal Gauss weights',
                  'initial':'dimensionless training-prior initial-RMS consistency; stationary cases contribute zero'}
        if detailed:
            diagnostics['boundaries']={'prediction':tensor_summary(output.predicted_value,batch.queries.valid),'coefficients':tensor_summary(output.xi_hat,output.xi_active)}
            if self.architecture=='A':diagnostics['memory']={'tokens':tensor_summary(output.context.memory.tokens),'attention':tensor_summary(output.context.memory.attention)}
            else:diagnostics['interaction']={'tokens':tensor_summary(output.context.interaction_tokens),'attention':tensor_summary(output.context.attention)}
        return LossReport(losses['total'],raw,weighted,{k:meanings[k] for k in raw},counts,diagnostics)


def tensor_summary(tensor,valid=None):
    t=(tensor if valid is None else tensor[valid]).detach()
    finite=torch.isfinite(t);x=t[finite]
    return {'shape':list(tensor.shape),'dtype':str(tensor.dtype),'device':str(tensor.device),'nonfinite':int((~finite).sum()),
            'min':float(x.min()) if x.numel() else None,'max':float(x.max()) if x.numel() else None,
            'mean':float(x.mean()) if x.numel() else None,'std':float(x.std(unbiased=False)) if x.numel() else None}


def capability_probe(model,architecture,normalizers,device):
    """Actual scientific forward/parameter-gradient probe BEFORE any optimizer step.

    No .backward, parameter-gradient mutation, global RNG, device fallback or model
    simplification. C additionally exercises independent strong second derivatives.
    """
    from .a_checks import mixed_fixture,slice_tokens
    from .constraints import strong_diagnostics
    previous=model.training
    try:
        model.eval()
        with torch.inference_mode(False),torch.enable_grad():
            batch=mixed_fixture(normalizers,device,queries=2)
            adapter=ScientificAdapter(architecture);report=adapter.loss(model,batch)
            grads=torch.autograd.grad(report.total,tuple(model.parameters()),allow_unused=True)
            require(bool(torch.isfinite(report.total)) and all(g is None or bool(torch.isfinite(g).all()) for g in grads),'capability probe produced nonfinite loss/derivatives')
            if architecture=='C':
                context=model.prepare_context(batch.specs,batch.observations)
                strong_diagnostics(model,context,batch.queries)
        return {'passed':True,'device':device,'neural_dtype':'float32','coordinate_second_derivatives':architecture=='C','physical_bridge':architecture in {'B0','B1'}}
    except Exception as error:
        raise RuntimeError(f'{architecture} capability probe failed on {device}/float32, operation: {type(error).__name__}: {error}; no fallback performed') from error
    finally:model.train(previous)

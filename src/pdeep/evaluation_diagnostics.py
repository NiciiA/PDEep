"""Controlled observation experiments and non-training comparison protocols."""
from dataclasses import dataclass, replace, fields
import torch
from .batches import ObservationBatch
from .data import array_hash
from .core import digest, require
from .evaluation_inference import slice_query


def input_identity(batch):return digest({f.name:array_hash(getattr(batch,f.name)) for f in fields(batch)})

def observation_variants(obs, pde):
    """Clone before changing; positions/times/quantity manipulations hold y/sigma.

    Sensor movement is deliberately a counterfactual inconsistent observation,
    not regenerated physical data. Whole-record permutation is a different test.
    """
    def clone():return ObservationBatch(**{f.name:getattr(obs,f.name).clone() for f in fields(obs)})
    def records(order):return ObservationBatch(**{f.name:getattr(obs,f.name)[:,order].clone() for f in fields(obs)})
    n=obs.valid.shape[1]; order=torch.arange(n-1,-1,-1)
    subset=clone();subset.valid[:,::2]=False
    sigma=clone();sigma.sigma.mul_(2)
    value=clone();value.value[:,0]+=3*value.sigma[:,0]
    moved=clone();moved.position[:,0,0]=.05+.9*(1-(moved.position[:,0,0]-.05)/.9)
    reassigned=clone();reassigned.value[:]=obs.value[:,order]
    variants={'remove_even_records':(subset,'half invalid; remaining y/locations/time/quantity/sigma fixed'),
              'sigma_x2':(sigma,'all sigma doubled; y/locations/time/quantity fixed'),
              'change_first_value':(value,'first y += 3 original sigma; other metadata fixed'),
              'move_first_sensor':(moved,'first x reflected in [.05,.95]; y/sigma/time/quantity fixed'),
              'record_permutation':(records(order),'reverse complete records; same physical observation set'),
              'value_reassignment':(reassigned,'reverse values only; positions/sigma/time/quantity fixed'),
              'empty':(records(torch.arange(0)), 'no measurements; D returns prior exactly')}
    if pde!='poisson':
        timed=clone();timed.physical_time[:,0]=.37
        variants['change_first_time']=(timed,'first physical t=.37; x/y-value/sigma/quantity fixed')
    if pde=='wave':
        quantity=clone();quantity.quantity_id[:,0]=1-quantity.quantity_id[:,0]
        variants['flip_first_quantity']=(quantity,'first u<->v; value/sigma/position/time fixed; counterfactual metadata')
    return variants


def observation_diagnostics(method, spec, obs, probe, baseline=None):
    original=input_identity(obs);base=method.prepare(spec,obs) if baseline is None else baseline
    pred=base.predict(probe);result={}
    for name,(changed,semantics) in observation_variants(obs,spec.name).items():
        changed.validate((spec,));session=method.prepare(spec,changed);other=session.predict(probe)
        row={'semantics':semantics,'input_identity':input_identity(changed),'valid_sensors':int(changed.valid.sum()),
             'field_max_change':float((other-pred).abs().max()),'coefficient_L2_change':float((session.xi-base.xi).norm())}
        if method.architecture=='D':
            row['posterior_covariance_trace']=float(session.context.covariance.trace())
            row['covariance_trace_change']=float(session.context.covariance.trace()-base.context.covariance.trace())
            if name in {'sigma_x2','remove_even_records','empty'}:
                row['covariance_difference_min_eigenvalue']=float(torch.linalg.eigvalsh(session.context.covariance-base.context.covariance).min())
        result[name]=row
    require(input_identity(obs)==original,'observation diagnostic mutated original plan data')
    return {'experiments':result,'original_unchanged':True,'interpretation':'neural input-path sensitivity is not evidence of physical correctness'}

@dataclass(frozen=True)
class ComparisonProtocol:
    protocol: str='matched_successful_updates'
    successful_updates: int | None=10000
    wall_clock_seconds: float | None=None
    seeds: tuple[int, ...]=(20261108,20261109,20261110)
    equation_encoding: str='typed_graph'
    label_only_target_parameters: int | None=None
    capacity_tolerance_fraction: float=.01

    def validate(self, *, serious=False):
        require(self.protocol in {'matched_successful_updates','matched_wall_clock'},'unknown comparison protocol')
        if self.protocol=='matched_successful_updates':require(type(self.successful_updates) is int and self.successful_updates>0 and self.wall_clock_seconds is None,'declare only successful-update budget')
        else:require(self.successful_updates is None and self.wall_clock_seconds is not None and self.wall_clock_seconds>0,'declare only wall-clock budget')
        require(len(set(self.seeds))==len(self.seeds) and (not serious or len(self.seeds)>=3),'serious results need >=3 independent seeds')
        require(self.equation_encoding in {'typed_graph','label_only'} and 0<=self.capacity_tolerance_fraction<=.1,'invalid equation ablation config')
        require(self.equation_encoding!='label_only' or (self.label_only_target_parameters is not None and self.label_only_target_parameters>0),'later label-only implementation requires explicit matched capacity target')
        return self


def equation_structure_hook(method, spec, observations, *, reference_spec=None):
    """Bounded structural probe; parsing/encoding success makes NO solver claim.

    Callers supply validated changed numeric bindings, status masks, equivalent
    DAGs or unsupported structures. B/D retain strict canonical solver support.
    This cannot produce benchmark scores for a changed physical problem.
    """
    try:
        spec.validate()
        if method.model is None:
            from .reference import require_supported
            require_supported(spec)
            return {'accepted':True,'scope':'canonical physical solver only'}
        from .encoding import pack_graphs
        from .representations import float32_batch
        with torch.no_grad():
            # Encoder is independently owned by each architecture.
            encoder=method.model.equation
            graph=encoder(pack_graphs((spec,),method.model.vocabulary).to(method.device))
        return {'accepted':True,'scope':'graph encoder only; no changed-physics reconstruction claim',
                'context_norm':float(graph.context.norm()),'context_identity':array_hash(graph.context),
                'physical_solver_supported':spec == __import__('pdeep.reference',fromlist=['canonical_spec']).canonical_spec(spec.name)}
    except (ValueError,RuntimeError,AttributeError) as exc:
        return {'accepted':False,'reason':str(exc),'scope':'explicit structural rejection'}

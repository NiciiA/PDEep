"""Inference-only adapters. Scoring truth cannot enter these interfaces."""
from dataclasses import dataclass, field, asdict
from pathlib import Path
from types import SimpleNamespace
import json
import time
import torch
from .core import require, digest
from .batches import QueryBatch
from .representations import float32_batch
from .b_physics import build_system, fixed_refinement, physical_solution
from .bayes import ExactGaussianD, modal_prior, sensor_matrix
from .architecture_b import BAblation
from .train_validation import model_identity

STATUSES=('untrained','smoke','bounded','partially trained','trained','exact; no training')


def sync(device):
    device=torch.device(device)
    if device.type=='mps':torch.mps.synchronize()
    elif device.type=='cuda':torch.cuda.synchronize(device)


def slice_query(query, lo, hi):
    return QueryBatch(*(getattr(query,n)[:,lo:hi] for n in ('position','physical_time','time_present','quantity_id','valid')))

@dataclass
class Method:
    name: str
    architecture: str
    status: str
    model: object=None
    checkpoint: str='untrained initialization'
    training: dict=field(default_factory=dict)
    ablation: BAblation=field(default_factory=BAblation)

    def __post_init__(self):
        require(self.status in STATUSES,'explicit supported training status required')
        require(self.architecture in {'A','C','B0','B1','D','Fixed','Prior'},'unsupported evaluation method')
        require((self.model is not None)==(self.architecture in {'A','C','B0','B1'}), 'model/architecture mismatch')

    @property
    def device(self):return next(self.model.parameters()).device if self.model is not None else torch.device('cpu')

    def metadata(self):
        return {'architecture':self.architecture,'training_status':self.status,'checkpoint':self.checkpoint,
                'training':self.training,'device':str(self.device),'precision':'float32 neural; explicit CPU64 B physics' if self.model is not None else 'CPU float64',
                'parameters':sum(p.numel() for p in self.model.parameters()) if self.model is not None else 0,
                'parameter_bytes':sum(p.numel()*p.element_size() for p in self.model.parameters()) if self.model is not None else 0,
                'ablation':asdict(self.ablation) if self.architecture.startswith('B') else None,
                'posterior_uncertainty':'exact Gaussian' if self.architecture=='D' else 'unavailable'}

    def prepare(self, spec, observations):
        """Only known physics and measured inputs. No case, xi_true or clean values."""
        observations.validate((spec,))
        with torch.no_grad():
            if self.architecture=='D':
                state=ExactGaussianD().fit(spec,observations);xi=state.mean
            elif self.architecture=='Prior':state=modal_prior(spec);xi=state.mean
            elif self.architecture=='Fixed':
                state=build_system((spec,),observations)
                trajectory,objectives=fixed_refinement(state,steps=3);xi=trajectory[0,-1,:18 if spec.name=='wave' else 9]
                state=SimpleNamespace(system=state,trajectory=trajectory,objectives=objectives)
            else:
                context=self.model.prepare_context((spec,),float32_batch(observations,device=self.device.type if self.device.type=='mps' else str(self.device)))
                if self.architecture.startswith('B'):
                    state=self.model.infer_unknowns(context,ablation=self.ablation,return_diagnostics=True)
                    xi=state.xi_hat[0,:18 if spec.name=='wave' else 9]
                else:
                    state=context;xi=self.model.infer_unknowns(context).xi_hat[0,:18 if spec.name=='wave' else 9]
        return Session(self,spec,state,xi.detach().cpu().double())

@dataclass
class Session:
    method: Method
    spec: object
    context: object
    xi: torch.Tensor

    def predict(self, query, *, chunk=1024):
        require(type(chunk) is int and chunk>0, 'positive query chunk required')
        query.validate((self.spec,)); out=[]
        with torch.no_grad():
            for lo in range(0,query.valid.shape[1],chunk):
                q=slice_query(query,lo,lo+chunk)
                if self.method.architecture=='D':value=self.context.predict(q).mean
                elif self.method.architecture in {'Fixed','Prior'}:
                    value=(sensor_matrix(self.spec,q)@self.xi)[None,:,None]
                elif self.method.architecture.startswith('B'):
                    value=self.method.model.predict_queries(self.context,q).physical_value
                else:
                    device=self.method.device;qn=float32_batch(q,device=device.type if device.type=='mps' else str(device))
                    result=self.method.model.predict_queries(self.context,qn)
                    # B's physical_value retains the actual CPU64 modal solution.
                    value=result.physical_value if self.method.architecture.startswith('B') else result.predicted_value
                out.append(value.detach().cpu().double())
        return torch.cat(out,dim=1) if out else torch.zeros(1,0,1,dtype=torch.float64)


def available_methods(root, *, device='cpu', include_controls=True):
    """Reload checkpoints without capability probes, backward, or optimizer steps.

    Fork CPU RNG because model construction initializes weights before loading.
    Checkpoint metadata remains authoritative; completion of smoke != trained.
    """
    from .train_checkpoint import load_checkpoint
    from .diagnostics import resolve_device
    resolved,_=resolve_device(device,"float32")
    root=Path(root); pointer=json.loads((root/'reports/latest_training_engine.json').read_text())
    report=json.loads((root/pointer['report']).read_text()); methods=[]; norm=None
    for name in ('A','C','B0','B1'):
        record=report['records'].get(name)
        if record is None:continue
        manifest=record['state']['last_checkpoint']
        with torch.random.fork_rng(devices=[]):
            engine=load_checkpoint(manifest,expected_architecture=name,probe=False)
        if norm is None:norm=engine.normalizers
        require(norm.identity==engine.normalizers.identity,'checkpoint normalizers disagree')
        c=engine.config;s=engine.state
        status='untrained' if s.successful==0 else c.preset if c.preset in {'smoke','bounded'} else 'trained' if s.successful>=c.updates else 'partially trained'
        train={'successful_updates':s.successful,'declared_updates':c.updates,'effective_batch':c.effective_batch,
               'cases_processed':s.cases_processed,'root_seed':c.root_seed,'device':c.device,'source_identity':c.source_id,
               'config_identity':c.identity,'elapsed_seconds_including_overhead':s.elapsed_seconds,
               'validation_seconds':s.validation_seconds,'saving_seconds':s.saving_seconds,
               'successful_update_timings':s.recent_steps,
               'median_successful_update_seconds':__import__('statistics').median(sum(row['timings'].get(k,0) for k in ('data','forward','backward','optimizer')) for row in s.recent_steps) if s.recent_steps else None,
               'C_derivative_training_cost':None}
        if name=='C':
            train['C_derivative_training_cost']={
                'selected_checkpoint_isolated_derivative_seconds':None,
                'reason':'Stage7A records combined forward/backward timings, not an isolated coordinate-derivative timer',
                'historical_run':None}
            previous_pointer=root/'reports/latest_architecture_C.json'
            if previous_pointer.exists():
                previous_meta=json.loads(previous_pointer.read_text())
                previous=json.loads((root/previous_meta['report']).read_text())
                train['C_derivative_training_cost']['historical_run']={
                    'report':previous_meta['report'],'run_id':previous['run_id'],
                    'device':previous['training']['device'],
                    'scope':'different historical C checkpoint; excluded from matched training-cost comparison',
                    'updates':[{k:row[k] for k in ('coordinate_derivative_seconds','quadrature_forward_seconds','forward_seconds','backward_seconds')}
                               for row in previous['training']['updates']]}
        model=engine.model.to(resolved);model.eval()
        methods.append(Method(name,name,status,model,str(manifest),train))
    require(norm is not None,'no available checkpoints; construct Method with explicit untrained models/normalizers')
    methods.append(Method('D','D','exact; no training',checkpoint='exact-T1-Gaussian'))
    if include_controls:
        b=next((m for m in methods if m.name=='B1'),None)
        if b:
            # Evaluation-only views of the SAME checkpoint, contexts freshly built.
            methods.extend([Method('B1-no-spatial','B1',b.status,b.model,b.checkpoint,b.training,BAblation(spatial=False)),
                            Method('B1-no-attention','B1',b.status,b.model,b.checkpoint,b.training,BAblation(spatial_attention=False))])
        methods.extend([Method('Fixed','Fixed','exact; no training',checkpoint='prior start, D=S^2, eta=1/L, 3 steps'),
                        Method('Prior','Prior','exact; no training',checkpoint='modal prior mean')])
    return methods,norm

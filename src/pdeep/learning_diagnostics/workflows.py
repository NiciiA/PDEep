"""Review, read-only preflight and direct synchronous opt-in diagnostic execution."""
from dataclasses import dataclass,asdict
from pathlib import Path
from datetime import datetime,timezone
import json
import torch
from ..core import digest,require
from ..artifact_io import read_manifest,atomic_json
from ..artifact_drive import PersistencePolicy
from ..artifact_tensors import state_identity
from ..geometric.models import create
from ..smooth.training import select_device
from .data import prepare_data,EXPERIMENTS
from .engine import DiagnosticConfig,DiagnosticEngine,resume_diagnostic,FieldOnly
from .dashboard import Dashboard
from .metrics import reconstruction,representation,gradient_terms,physical_diagnostics
from .sensitivity import sensitivity

@dataclass
class DiagnosticPlan:
    configs: tuple
    data: object
    models: tuple
    certificate: dict
    action: str
    checkpoints: tuple
    root: str
    persistence: object
    settings: dict
    initial_model_ids: tuple
    @property
    def identity(self):
        return digest((tuple(asdict(c) for c in self.configs),self.data.identity,self.certificate['certification_id'],self.action,self.checkpoints,self.root,self.persistence.identity,self.settings,self.initial_model_ids))


def prepare_diagnostic(*,model='A_GEO',experiment='poisson_square',capacity='medium',updates=500,device='auto',seed=42,action='fresh',certificate,root='artifacts',persistence=None,checkpoints=None,run_name=None,advanced=None):
    require(model in {'A_GEO','C_GEO'} and experiment in EXPERIMENTS,'select a real geometry model and one supported diagnostic')
    require(action in {'fresh','resume'},'diagnostic action fresh/resume only')
    require(experiment!='c_physics_ablation' or model=='C_GEO','paired physics ablation requires C_GEO')
    advanced=dict(advanced or {});allowed={'pde','family','query_count','grid_size','conditioning_h','learning_rate','weight_decay','scheduler','warmup_updates','validation_every','checkpoint_every','instrumentation_every'}
    require(set(advanced)<=allowed,'unknown advanced diagnostic settings: '+str(set(advanced)-allowed))
    data=prepare_data(Path(root)/'learning_diagnostics'/'fixed_data',certificate,experiment=experiment,seed=seed,count=advanced.get('query_count',128),grid_size=advanced.get('grid_size',64),conditioning_h=advanced.get('conditioning_h',.12),pde=advanced.get('pde'),family=advanced.get('family'))
    objectives=('FULL','FIELD_ONLY') if experiment=='c_physics_ablation' else ('FULL',)
    configs=[];models=[];paths=[]
    run_name=run_name or 'diagnostic-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    for objective in objectives:
        path=(checkpoints or {}).get(objective) if isinstance(checkpoints,dict) else checkpoints
        if action=='resume':
            require(path is not None,'choose verified diagnostic checkpoint(s), keyed FULL/FIELD_ONLY for paired ablation')
            r=read_manifest(path,kind='training_checkpoint',architecture=model); d=r['config']
            require(r.get('scientific_schema')==__import__('pdeep.learning_diagnostics.engine',fromlist=['SCHEMA']).SCHEMA,'mixed-pool checkpoints are not implicit warm starts')
            d={**d,**{k:tuple(d[k]) for k in ('pdes','probabilities','training_families')}};c=DiagnosticConfig(**d)
            require((c.architecture,c.experiment,c.capacity,c.updates,c.device,c.root_seed,c.objective_variant,c.data_identity)==(model,experiment,capacity,updates,select_device(device),seed,objective,data.identity),'resume selected settings/data differ from saved experiment; restore saved settings (budget is not extended)')
            for key in set(advanced)-{'pde','family','grid_size'}:require(getattr(c,key)==advanced[key],f'resume {key} differs from checkpoint')
        else:
            opts={k:v for k,v in advanced.items() if k not in {'pde','family','grid_size'}}
            c=DiagnosticConfig(architecture=model,capacity=capacity,pdes=(data.case.pde,),probabilities=(1.,),training_families=(data.case.family,),cases_per_geometry=1,updates=updates,device=device,root_seed=seed,experiment=experiment,objective_variant=objective,data_identity=data.identity,bank_queries=max(128,opts.get('query_count',128)),run_tag=run_name+'-'+objective,**opts)
        # Preview builds real architecture without optimizer or hidden target inputs.
        m=create(model,data.normalizers,capacity,seed=seed,device=c.device)
        if path is not None and experiment=='observation_sensitivity' and action=='fresh':
            from .inspection import load_model
            m,_=load_model(path,certificate,device=c.device)
            require((m.config.architecture,m.config.capacity)==(model,capacity),'sensitivity checkpoint architecture/capacity differs from selection')
        configs.append(c);models.append(m);paths.append(None if path is None else str(path))
    if len(models)==2:
        require(state_identity(models[0])==state_identity(models[1]),'ablation initializations differ')
        require(all(a.data_ptr()!=b.data_ptr() for a,b in zip(models[0].parameters(),models[1].parameters())),'ablation instances must not share parameters')
    checkpoints=({k:str(v) for k,v in checkpoints.items()} if isinstance(checkpoints,dict) else None if checkpoints is None else str(checkpoints))
    settings=dict(model=model,experiment=experiment,capacity=capacity,updates=updates,device=device,seed=seed,action=action,checkpoints=checkpoints,advanced=advanced)
    return DiagnosticPlan(tuple(configs),data,tuple(models),certificate,action,tuple(paths),str(root),persistence or PersistencePolicy(),settings,tuple(state_identity(m) for m in models))


def diagnostic_preview(plan):
    return {'configs':[asdict(c) for c in plan.configs],'parameters':[m.parameter_count for m in plan.models],'independent_instances':True,'physical_case':plan.data.case.manifest,'identities':plan.data.identities,'training_query_policy':'strict fixed TRAIN case and main queries; independent diagnostic grid; NOT generalization','normalizers':plan.data.normalizers.values[plan.data.case.pde],'objective_weights':[{'field':1.,'auxiliary_supervision':.1,**({'weak':.1,'initial':.1,'boundary':.1} if c.architecture=='C_GEO' and c.objective_variant=='FULL' else {'weak':0.,'initial':0.,'boundary':0.})} for c in plan.configs],'optimizer':'existing AdamW + successful-update scheduler; constant LR default','source_revision':'config.source_id + diagnostic_source_id; exact resume checked','certificate':plan.certificate['certification_id'],'optimizer_updates':0,'checkpoints':plan.checkpoints,'reviewed_identity':plan.identity}


def preflight(plan,*,loss_gradients=False,observations=False):
    from ..geometric.physics import Adapter,WeakConfig
    result={}
    for c,m in zip(plan.configs,plan.models):
        before=state_identity(m); r={'reconstruction':reconstruction(m,plan.data),'representation':representation(m,plan.data)}
        if loss_gradients:
            adapter=(FieldOnly if c.objective_variant=='FIELD_ONLY' else Adapter)(c.architecture,WeakConfig(c.spatial_order,c.time_order,c.weak_tests,c.physics_chunk))
            r['loss_gradients']=gradient_terms(m,plan.data.batch(c.device),adapter)
            if c.architecture=='C_GEO':r['physical_diagnostics']=physical_diagnostics(m,plan.data,adapter.weak)
        if observations:r['sensitivity']=sensitivity(m,plan.data)
        require(before==state_identity(m),'non-optimizing preflight changed parameters')
        result[c.objective_variant]=r
    return result


def launch_diagnostic(plan,*,expected_identity,run_mode='preview',monitor=None):
    require(expected_identity==plan.identity,'diagnostic plan changed; rerun preview')
    require(run_mode in {'preview','train','resume'},'diagnostic mode preview/train/resume only')
    if run_mode=='preview':return {}
    require((run_mode=='resume')==(plan.action=='resume'),'diagnostic run mode/action mismatch')
    require(tuple(state_identity(m) for m in plan.models)==plan.initial_model_ids,'preview model parameters changed; construct a new plan')
    monitor=monitor or Dashboard();results={}
    for c,m,path in zip(plan.configs,plan.models,plan.checkpoints):
        if c.experiment=='observation_sensitivity':
            # Read-only inference diagnostics, intentionally never train this experiment.
            if path:
                from .inspection import load_model
                m,_=load_model(path,plan.certificate,device=c.device)
                require((m.config.architecture,m.config.capacity)==(c.architecture,c.capacity),'sensitivity checkpoint model/capacity differs')
            paired=prepare_data(Path(plan.root)/'learning_diagnostics'/'fixed_data',plan.certificate,experiment=c.experiment,seed=c.root_seed,count=c.query_count,grid_size=4,conditioning_h=c.conditioning_h,pde=plan.data.case.pde,family=plan.data.case.family,paired=True)
            result=sensitivity(m,plan.data,paired);dest=Path(plan.root)/'learning_diagnostics'/'observation_sensitivity'/c.run_id
            atomic_json(dest/'sensitivity.json',result);results[c.objective_variant]={'result':result,'path':str(dest),'optimizer_updates':0};continue
        if plan.action=='resume':engine=resume_diagnostic(path,plan.root,data=plan.data,certificate=plan.certificate,config=c,persistence=plan.persistence,monitor=monitor)
        else:engine=DiagnosticEngine(c,plan.root,data=plan.data,certificate=plan.certificate,model=m,persistence=plan.persistence,monitor=monitor)
        # Gradient conflicts measured once without changing the actual update.
        atomic_json(engine.run_dir/'gradient_terms.json',gradient_terms(engine.model,plan.data.batch(c.device),engine.adapter))
        engine.run()  # shared synchronous update/run; no background worker or callback launch
        atomic_json(engine.run_dir/'representation.json',representation(engine.model,plan.data))
        if c.architecture=='C_GEO':atomic_json(engine.run_dir/'physical_diagnostics.json',physical_diagnostics(engine.model,plan.data,engine.adapter.weak))
        engine.export();results[c.objective_variant]=engine
    return results

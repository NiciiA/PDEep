"""Review, read-only preflight and direct synchronous opt-in diagnostic execution."""
from dataclasses import dataclass,asdict,field,replace
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
    performance: dict=field(default_factory=dict)
    reference_cache: object=field(default=None,repr=False)
    @property
    def identity(self):
        return digest((tuple(asdict(c) for c in self.configs),self.data.identity,self.certificate['certification_id'],self.action,self.checkpoints,self.root,self.persistence.identity,self.settings,self.initial_model_ids))


def prepare_diagnostic(*,model='A_GEO',experiment='poisson_square',capacity='medium',updates=500,device='auto',seed=42,action='fresh',certificate,root='artifacts',persistence=None,checkpoints=None,run_name=None,advanced=None,query_mode=None,query_count=None,query_time_policy='stratified',query_sampler_seed=42,query_chunk=512,query_checkpointing=True,amp_mode='off',memory_target_gb=11.):
    require(model in {'A_GEO','C_GEO'} and experiment in EXPERIMENTS,'select a real geometry model and one supported diagnostic')
    require(action in {'fresh','resume'},'diagnostic action fresh/resume only')
    require(experiment!='c_physics_ablation' or model=='C_GEO','paired physics ablation requires C_GEO')
    advanced=dict(advanced or {});allowed={'pde','family','query_count','grid_size','conditioning_h','learning_rate','weight_decay','scheduler','warmup_updates','validation_every','checkpoint_every','instrumentation_every'}
    require(set(advanced)<=allowed,'unknown advanced diagnostic settings: '+str(set(advanced)-allowed))
    modern=query_mode is not None
    requested_count=query_count if query_count is not None else advanced.get('query_count',1024 if modern else 128)
    require(query_mode in {None,'fixed','resampled','dense_resampled'} and 2<=requested_count<=16384,'invalid query strategy/count')
    base_count=requested_count if query_mode in {None,'fixed'} else min(requested_count,1024)
    data=prepare_data(Path(root)/'learning_diagnostics'/'fixed_data',certificate,experiment=experiment,seed=seed,count=base_count,grid_size=advanced.get('grid_size',64),conditioning_h=advanced.get('conditioning_h',.12),pde=advanced.get('pde'),family=advanced.get('family'),cache_reference=modern)
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
            require(not modern or (c.query_mode,c.query_count,c.query_time_policy,c.query_sampler_seed,c.query_chunk,c.query_checkpointing,c.amp_mode,c.memory_target_gb)==(query_mode,requested_count,query_time_policy,query_sampler_seed,query_chunk,query_checkpointing,amp_mode,memory_target_gb),'resume query/precision/memory settings differ from checkpoint')
            for key in set(advanced)-{'pde','family','grid_size'}:require(getattr(c,key)==advanced[key],f'resume {key} differs from checkpoint')
        else:
            opts={k:v for k,v in advanced.items() if k not in {'pde','family','grid_size'}}
            opts['query_count']=requested_count
            if modern:opts.update(query_revision='continuous-queries-v2',query_mode=query_mode,query_time_policy=query_time_policy,query_sampler_seed=query_sampler_seed,query_chunk=query_chunk,query_checkpointing=query_checkpointing,amp_mode=amp_mode,memory_target_gb=memory_target_gb)
            c=DiagnosticConfig(architecture=model,capacity=capacity,pdes=(data.case.pde,),probabilities=(1.,),training_families=(data.case.family,),cases_per_geometry=1,updates=updates,device=device,root_seed=seed,experiment=experiment,objective_variant=objective,data_identity=data.identity,bank_queries=max(128,opts.get('query_count',128)),run_tag=run_name+'-'+objective,**opts)
        # Preview builds real architecture without optimizer or hidden target inputs.
        m=create(model,data.normalizers,capacity,seed=seed,device=c.device)
        if path is not None and (experiment=='observation_sensitivity' or action=='resume'):
            from .inspection import load_model
            m,_=load_model(path,certificate,device=c.device)
            require((m.config.architecture,m.config.capacity)==(model,capacity),'sensitivity checkpoint architecture/capacity differs from selection')
        configs.append(c);models.append(m);paths.append(None if path is None else str(path))
    if len(models)==2:
        if action=='fresh':require(state_identity(models[0])==state_identity(models[1]),'ablation initializations differ')
        require(all(a.data_ptr()!=b.data_ptr() for a,b in zip(models[0].parameters(),models[1].parameters())),'ablation instances must not share parameters')
    checkpoints=({k:str(v) for k,v in checkpoints.items()} if isinstance(checkpoints,dict) else None if checkpoints is None else str(checkpoints))
    settings=dict(model=model,experiment=experiment,capacity=capacity,updates=updates,device=device,seed=seed,action=action,checkpoints=checkpoints,advanced=advanced,query_mode=query_mode,query_count=requested_count,query_time_policy=query_time_policy,query_sampler_seed=query_sampler_seed,query_chunk=query_chunk,query_checkpointing=query_checkpointing,amp_mode=amp_mode,memory_target_gb=memory_target_gb)
    return DiagnosticPlan(tuple(configs),data,tuple(models),certificate,action,tuple(paths),str(root),persistence or PersistencePolicy(),settings,tuple(state_identity(m) for m in models),reference_cache=data.cached_reference)


def diagnostic_preview(plan):
    return {'configs':[asdict(c) for c in plan.configs],'parameters':[m.parameter_count for m in plan.models],'independent_instances':True,'physical_case':plan.data.case.manifest,'identities':plan.data.identities,'training_query_policy':plan.configs[0].query_mode+'; fixed TRAIN case/sensors and independent evaluation; no test calibration','normalizers':plan.data.normalizers.values[plan.data.case.pde],'objective_weights':[{'field':1.,'auxiliary_supervision':.1,**({'weak':.1,'initial':.1,'boundary':.1} if c.architecture=='C_GEO' and c.objective_variant=='FULL' else {'weak':0.,'initial':0.,'boundary':0.})} for c in plan.configs],'optimizer':'existing AdamW + successful-update scheduler; constant LR default','source_revision':'config.source_id + diagnostic_source_id; exact resume checked','certificate':plan.certificate['certification_id'],'optimizer_updates':0,'checkpoints':plan.checkpoints,'reviewed_identity':plan.identity}


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
    if any(c.query_revision!='legacy-fixed-v1' and c.device.startswith('cuda') for c in plan.configs) and not plan.performance:
        performance_preflight(plan)
        require(expected_identity==plan.identity,'precision resolved by CUDA pilot; rerun/review preview before training')
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
        engine.performance_preflight=plan.performance.get(c.objective_variant,{})
        if hasattr(engine.sampler,'reference') and plan.reference_cache is not None:engine.sampler.reference=plan.reference_cache
        # Gradient conflicts measured once without changing the actual update.
        atomic_json(engine.run_dir/'gradient_terms.json',gradient_terms(engine.model,plan.data.batch(c.device),engine.adapter))
        engine.run()  # shared synchronous update/run; no background worker or callback launch
        atomic_json(engine.run_dir/'representation.json',representation(engine.model,plan.data))
        if c.architecture=='C_GEO':atomic_json(engine.run_dir/'physical_diagnostics.json',physical_diagnostics(engine.model,plan.data,engine.adapter.weak))
        engine.export();results[c.objective_variant]=engine
    return results


def performance_preflight(plan,*,counts=None,optimizer_step=False,cuda_kernel_profile=False):
    from .performance import profile,resolve_amp
    from .queries import QuerySampler
    from .query_loss import QueryAdapter
    from ..geometric.physics import WeakConfig
    outcomes={};configs=[];reference_cache=plan.reference_cache
    for c,m in zip(plan.configs,plan.models):
        if c.query_revision=='legacy-fixed-v1':configs.append(c);continue
        sampler=QuerySampler(plan.data,c,reference_cache=reference_cache);sampler.measure_transfer=True
        batch,preparation=sampler.batch(0);reference_cache=sampler.reference
        adapter=QueryAdapter(c.architecture,WeakConfig(c.spatial_order,c.time_order,c.weak_tests,c.physics_chunk),chunk=c.query_chunk,checkpointing=c.query_checkpointing,field_only=c.objective_variant=='FIELD_ONLY')
        precision=({'resolved':'off','reason':'paired C physics ablation uses common FP32 for both objectives'} if len(plan.configs)>1 and c.amp_mode=='auto' else resolve_amp(m,batch,adapter,c.amp_mode))
        if plan.action=='resume':require(precision['resolved']==c.resolved_amp_mode,'resume AMP numerical pilot differs from saved precision; no silent migration')
        c=replace(c,resolved_amp_mode=precision['resolved']);adapter.amp=c.resolved_amp_mode
        rows=[]
        for n in counts or (c.query_count,):
            require(2<=n<=16384,'preflight counts must be2..16384')
            if n!=c.query_count:
                cfg=replace(c,query_count=n,bank_queries=max(c.bank_queries,n),query_mode='resampled')
                temp=QuerySampler(plan.data,cfg,reference_cache=reference_cache);temp.measure_transfer=True;bb,pp=temp.batch(0);reference_cache=temp.reference
            else:bb,pp=batch,preparation
            row=profile(m,bb,adapter,memory_target_gb=c.memory_target_gb,optimizer_step=optimizer_step,cuda_kernel_profile=cuda_kernel_profile);row['data_preparation']=pp;rows.append(row)
            if n==c.query_count:require(row['status']=='fits','configured query budget exceeds measured memory allowance; lower QUERY_COUNT/QUERY_CHUNK_SIZE or enable checkpointing')
        outcomes[c.objective_variant]={'config_identity':c.identity,'precision':precision,'profiles':rows,'model_state_identity':state_identity(m)};configs.append(c)
    plan.configs=tuple(configs);plan.performance=outcomes;plan.reference_cache=reference_cache
    return outcomes

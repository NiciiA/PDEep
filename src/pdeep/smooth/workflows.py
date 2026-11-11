"""Small reviewable plans; only an explicit launch call creates a training engine."""
from dataclasses import asdict,dataclass
from pathlib import Path
import time,json
from ..core import require,digest
from ..artifact_io import read_manifest
from .definition import BENCHMARK_ID,SEEN,PDES,manifest,POLICY,geometry
from .certification import load_record,authorize,normalizers
from .training import Config,Engine,resume,select_device

@dataclass(frozen=True)
class Plan:
    configs: tuple
    certificate: dict
    action: str
    checkpoints: tuple
    run_root: str
    persistence: object
    @property
    def identity(self):return digest((tuple(asdict(c) for c in self.configs),self.certificate['certification_id'],self.action,self.checkpoints,self.run_root,self.persistence.identity))


def prepare(*,models=('A_GEO',),benchmark=BENCHMARK_ID,capacity='medium',pdes=PDES,device='auto',experiment='bounded',updates=1000,action='fresh',resume_checkpoint=None,save_every=250,validate_every=250,output_dir='artifacts',persistence=None,run_name=None,advanced=None,certificate_path=None):
    from ..artifact_drive import PersistencePolicy
    require(models and len(set(models))==len(models) and set(models)<={'A_GEO','C_GEO'},'select independent A_GEO and/or C_GEO')
    require(benchmark==BENCHMARK_ID,'G1 rough remains blocked; select the separately versioned G1_smooth_v1')
    require(action in {'fresh','resume'},'ACTION must be fresh or resume')
    certificate=load_record(certificate_path);authorize(certificate,benchmark=benchmark,pdes=pdes,families=SEEN)
    configs=[];paths=[]
    run_name=run_name or 'manual-'+time.strftime('%Y%m%d-%H%M%S')
    for name in models:
        path=resume_checkpoint.get(name) if isinstance(resume_checkpoint,dict) else resume_checkpoint
        if action=='resume':
            require(path is not None,'select a verified local/Drive checkpoint manifest for resume')
            r=read_manifest(path,kind='training_checkpoint',architecture=name)
            require(r.get('benchmark')==BENCHMARK_ID,'checkpoint benchmark mismatch; no old G1/T1 reinterpretation')
            kwargs=r['config'];kwargs={**kwargs,**{k:tuple(kwargs[k]) for k in ('pdes','probabilities','training_families')}};c=Config(**kwargs)
            require(c.capacity==capacity and tuple(c.pdes)==tuple(pdes) and c.device==select_device(device),'resume architecture/capacity/PDE/device configuration mismatch; select saved configuration')
            require(r['certificate']['certification_id']==certificate['certification_id'],'resume selected a different certification record')
        else:
            tiny=experiment=='tiny_overfit';families=('square','rectangle') if tiny else SEEN
            opts=dict(architecture=name,capacity=capacity,pdes=tuple(pdes),probabilities=tuple(1/len(pdes) for _ in pdes),device=device,preset=experiment,updates=updates,checkpoint_every=save_every,validation_every=validate_every,training_families=families,cases_per_geometry=1 if tiny else None,run_tag=run_name)
            opts.update(advanced or {});c=Config(**opts)
        configs.append(c);paths.append(None if path is None else str(Path(path)))
    return Plan(tuple(configs),certificate,action,tuple(paths),str(output_dir),persistence or PersistencePolicy())


def preview(plan):
    n=normalizers(plan.certificate)
    import numpy as np
    def mesh_sizes(c):
        result={}
        for family in c.training_families:
            m=geometry(family,c.conditioning_h);interior=len(m.vertices)-len(np.unique(m.boundary_edges))
            result[family]={'mesh_id':m.mesh_id,'vertices':len(m.vertices),'cells':len(m.cells),'interior':interior,'coarse_anchors':min(64,interior),'fine_anchors':min(256,interior)}
        return result
    counts={('A_GEO','medium'):6158789,('A_GEO','large'):10915077,('C_GEO','medium'):6455429,('C_GEO','large'):11441669}
    return [{'architecture':c.architecture,'benchmark':c.benchmark,'capacity':c.capacity,'parameter_count':counts.get((c.architecture,c.capacity),'verification fixture'),'independent_trainable_instance':True,'conditioning_meshes_and_actual_anchors':mesh_sizes(c),'device':c.device,'dtype':c.precision,'experiment':c.preset,'successful_updates':c.updates,'case_pool':c.case_pool_size,'case_sampling':'full lazy train split' if c.cases_per_geometry is None else 'explicit finite diagnostic pool','PDEs':c.pdes,'geometry_instances':c.training_families,'fixed_observation_set':0,'query_bank_per_case':c.bank_queries,'training_queries_per_case':c.query_count,'validation_grid':f'{c.validation_grid}x{c.validation_grid} domain-clipped x9 dynamic times xquantity','validation_scope':'same TRAIN cases; memorization only' if c.preset=='tiny_overfit' else 'paired unseen physical cases on seen geometries','reference_policy':POLICY,'conditioning_h':c.conditioning_h,'real_loss':'field + .1 auxiliary'+(' + .1 weak + .1 initial + .1 boundary' if c.architecture=='C_GEO' else ''),'certification_id':plan.certificate['certification_id'],'certification_status':plan.certificate['pass_fail_status'],'normalization_id':n.identity,'local_checkpoints':str(Path(plan.run_root)/BENCHMARK_ID/'runs'/c.architecture/c.run_id/'checkpoints'),'Drive_sync':plan.persistence.drive_enabled,'resume':p,'training_status':'pending user execution; constructing preview performs0 optimizer updates'} for c,p in zip(plan.configs,plan.checkpoints)]


def launch(plan,*,expected_identity=None,notebook=True):
    require(expected_identity==plan.identity,'reviewed plan changed; rerun preview and render a new launch button')
    from ..train_monitor import LiveMonitor
    results={}
    for c,path in zip(plan.configs,plan.checkpoints):
        monitor=LiveMonitor(enabled=True,notebook=notebook)
        if plan.action=='resume':engine=resume(path,plan.run_root,device=c.device,persistence=plan.persistence,monitor=monitor)
        else:engine=Engine(c,plan.run_root,certificate=plan.certificate,persistence=plan.persistence,monitor=monitor)
        engine.run();results[c.architecture]=engine
    return results


class EvaluationEngine(Engine):
    def update(self,*args,**kwargs):raise RuntimeError('Evaluation-only instance cannot optimize; use explicit exact resume for training.')
    def run(self,*args,**kwargs):raise RuntimeError('Evaluation-only instance cannot launch training.')
    def save_checkpoint(self):raise RuntimeError('Evaluation-only instance cannot write resumable training artifacts.')


def evaluate_checkpoint(path,output_dir,*,geometry_held_out=False,device=None,diagnostics=True):
    """Portable inference, explicitly distinct from same-device exact resume."""
    from ..artifact_io import read_payload,atomic_json
    from ..geometric.models import GeometryModel,ModelConfig
    from .training import SCHEMA,implementation_identity
    metadata,payload=read_payload(path,kind='training_checkpoint',required=('model','model_config','normalizers','training_state'))
    require(metadata.get('scientific_schema')==SCHEMA,'select a G1_smooth_v1 training checkpoint')
    require(metadata['implementation_identity']==implementation_identity(),'model/loss implementation changed; compatible inference loader required')
    cfg=metadata['config'];cfg={**cfg,**{k:tuple(cfg[k]) for k in ('pdes','probabilities','training_families')}}
    original_device=cfg['device'];cfg['device']=select_device('auto' if device is None else device)
    config=Config(**cfg);n=normalizers(metadata['certificate'])
    model=GeometryModel(ModelConfig(**payload['model_config']),n).to(config.device);model.load_state_dict(payload['model'],strict=True)
    engine=EvaluationEngine(config,Path(output_dir)/'smooth_evaluation'/metadata['artifact_id'],certificate=metadata['certificate'],model=model,allow_existing_run=True)
    engine.state.successful=payload['training_state']['successful']
    report=engine.validate(geometry_held_out=geometry_held_out,diagnostics=diagnostics)
    atomic_json(engine.run_dir/('evaluation-held-out.json' if geometry_held_out else 'evaluation-seen.json'),{**asdict(report),'checkpoint_manifest':str(path),'saved_device':original_device,'inference_device':config.device,'portable_inference':original_device!=config.device,'exact_training_resume':False})
    return engine,report


def summarize(report):
    import numpy as np
    rows=report.architecture_metrics['per_case_quantity']
    return rows

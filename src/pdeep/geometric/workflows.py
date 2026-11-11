"""Opt-in concise notebook workflows, no setup/optimizer side effects on import."""
from dataclasses import asdict
from pathlib import Path
import json,time
import numpy as np
import torch
from ..core import require
from .data import PDES,CATALOG,Normalizers,calibrate,physical_case,batch,rng
from .models import create
from .training import Config,Engine,resume


def normalizers(root):
    path=Path(root)/'geometry'/ 'normalizers.json';path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():return Normalizers.fromdict(json.loads(path.read_text()))
    n=calibrate(10,256);path.write_text(json.dumps(n.asdict(),indent=2));return n


def prepare(root,*,architecture='A_GEO',capacity='medium',device='cpu',updates=3,preset='smoke',tag='geometry',pdes=PDES,**advanced):
    from ..runtime import resolve_device
    resolve_device(device,'float32')
    c=Config(architecture=architecture,capacity=capacity,device=device,updates=updates,preset=preset,run_tag=tag,pdes=tuple(pdes),probabilities=tuple(1/len(pdes) for _ in pdes),**advanced)
    n=normalizers(root);m=create(architecture,n,capacity,seed=c.root_seed,device=device)
    return {'config':c,'normalizers':n,'model':m,'root':str(root),'preview':{'architecture':architecture,'capacity':capacity,'trainable_parameters':m.parameter_count,
            'inventory':m.inventory(),'config':asdict(c),'normalizer_identity':n.identity,'benchmark':'G1; independent of T1',
            'reference_warning':'Numerical G1 references have a finite mesh/time error budget. Inspect geometry_phase report before research training.',
            'launch':'No updates yet. Set GEO_RUN_MODE=train in the explicit launch cell.'}}


def launch(plan,*,mode='preview',persistence=None,resume_checkpoint=None):
    require(mode in {'preview','train','resume'},'explicit geometry run mode required')
    if mode=='preview':return None
    if mode=='resume':
        require(resume_checkpoint is not None,'select verified A_GEO/C_GEO checkpoint manifest')
        engine=resume(resume_checkpoint,Path(plan['root'])/'geometry'/'runs',device=plan['config'].device,persistence=persistence)
        require(engine.config.architecture==plan['config'].architecture and engine.config.capacity==plan['config'].capacity and engine.normalizers.identity==plan['normalizers'].identity,'selected geometry resume checkpoint incompatible with notebook selection')
    else:engine=Engine(plan['config'],plan['normalizers'],Path(plan['root'])/'geometry'/'runs',model=plan['model'],persistence=persistence)
    from ..train_monitor import LiveMonitor
    engine.monitor=LiveMonitor(enabled=True)
    engine.run(launch_full=engine.config.full_training_enabled);return engine


def explore(output_root,*,pde='wave',split='geometry_held_out',index=0):
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    n=normalizers(output_root);c=physical_case(split,pde,index);m=c.problem.mesh;o,meta=__import__('pdeep.geometric.data',fromlist=['token_set']).token_set(c,32,n,observation=True)
    from .representation import layout
    lay=layout(m);tri=mtri.Triangulation(*m.vertices.T,m.cells)
    fig,axes=plt.subplots(1,3,figsize=(14,4),constrained_layout=True)
    axes[0].triplot(tri,color='gray',lw=.3);axes[0].scatter(*m.vertices[lay.coarse].T,s=18,label='coarse');axes[0].scatter(*m.vertices[lay.fine].T,s=3,label='fine');axes[0].legend();axes[0].set_title('Conforming mesh and geodesic FPS anchors')
    t=None if pde=='poisson' else np.full(len(m.vertices),.237)
    value=c.evaluate(m.vertices,t);im=axes[1].tripcolor(tri,value,shading='gouraud');fig.colorbar(im,ax=axes[1]);axes[1].scatter(*o.position[0].numpy().T,c='k',s=8);axes[1].set_title('Reference u at physical t=.237 + sensors' if t is not None else 'Reference u + sensors')
    name='f' if pde=='poisson' else 'u0';v=c.fields[name](m.vertices);im=axes[2].tripcolor(tri,v,shading='gouraud');fig.colorbar(im,ax=axes[2]);axes[2].set_title('Hidden '+name+' (training target only)')
    for ax in axes:ax.set_aspect('equal')
    return c,o,fig,{'case':c.manifest,'observations':meta['identity'],'anchor_counts':lay.counts,'reference_device':'CPU float64','neural_shapes':'[B,N,2], [B,Q,2], [V,d], [M,d]; no modal xi head'}


def evaluate(checkpoint,output_root,*,splits=('validation','interpolation','geometry_held_out'),indices=(0,1),pdes=PDES,device='cpu'):
    """Paired accuracy/coverage/observation controls, aggregated without plot spam.

    Hidden targets are used only for scoring. Strong residual diagnostics are
    intentionally excluded on non-C2 mesh interfaces; use weak validation.
    """
    from .training import resume
    engine=resume(checkpoint,Path(output_root)/'geometry'/'evaluation_scratch',device=device);model=engine.model;model.eval();n=engine.normalizers;rows=[]
    with torch.no_grad():
        for split in splits:
            for pde in pdes:
                for i in indices:
                    c=physical_case(split,pde,int(i),h=engine.config.reference_h,dt=engine.config.reference_dt);b=batch([c],n,queries=256,set_id=724,device=device)
                    if device=='cuda':torch.cuda.synchronize()
                    elif device=='mps':torch.mps.synchronize()
                    start=time.perf_counter();pred=model(b.specs,b.observations,b.queries)
                    if device=='cuda':torch.cuda.synchronize()
                    elif device=='mps':torch.mps.synchronize()
                    seconds=time.perf_counter()-start
                    quantities={}
                    for q in (0,1) if pde=='wave' else (0,):
                        take=b.queries.quantity_id[0]==q;e=(pred[0,take]-b.queries.value[0,take]);truth=b.queries.value[0,take]
                        quantities['v' if q else 'u']={'physical_RMSE':float(e.square().mean().sqrt()),'standardized_MSE':float((e/n.scale(pde,q)).square().mean()),'relative_L2':float(e.norm()/truth.norm().clamp_min(1e-12))}
                    empty=batch([c],n,sensors=0,queries=256,set_id=724,device=device);prior=model(empty.specs,empty.observations,empty.queries)
                    rows.append({'split':split,'pde':pde,'case_id':c.case_id,'geometry_id':c.problem.mesh.geometry_id,'mesh_id':c.problem.mesh.mesh_id,
                        'geometry_family':c.problem.mesh.shape.family,'quantities':quantities,'inference_seconds':seconds,
                        'observation_effect_RMS':float((pred-prior).square().mean().sqrt()),'N':int(b.observations.valid.sum()),'Q':256,
                        'reference_budget':c.reference})
    grouped=[]
    for split in splits:
        for pde in pdes:
            group=[r for r in rows if r['split']==split and r['pde']==pde]
            grouped.append({'split':split,'pde':pde,'cases':len(group),'standardized_field_MSE':float(np.mean([np.mean([v['standardized_MSE'] for v in r['quantities'].values()]) for r in group])),
                           'median_seconds':float(np.median([r['inference_seconds'] for r in group]))})
    result={'benchmark':'G1','architecture':engine.config.architecture,'checkpoint':str(checkpoint),'rows':rows,'summary':grouped,'status':'bounded evaluation, not evidence of trained quality'}
    path=Path(output_root)/'geometry'/'evaluation.json';path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2));return result

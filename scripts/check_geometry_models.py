"""Opt-in Phase G bounded verification: <=3 actual updates per architecture.

Not run by pytest or notebook Run All. Writes only new geometry_phase artifacts.
The actual medium models receive full adapter backward checks; the large models
are instantiated/counted only. No research training or old checkpoints modified.
"""
from dataclasses import asdict,replace
from pathlib import Path
import argparse,json,time,tempfile,gc
import numpy as np
import torch
from pdeep.core import require
from pdeep.artifact_tensors import state_identity
from pdeep.artifact_metadata import scientific_identity
from pdeep.runtime import environment_report
from pdeep.geometric.data import Normalizers,calibrate,physical_case,batch,PDES,rng
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter,WeakConfig,weak_loss
from pdeep.geometric.representation import layout,retrieval_weights,interpolate
from pdeep.geometric.training import Engine,Config,resume


def run(device='cpu',updates=False):
    directory=Path('reports/geometry_phase');directory.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    require(json.loads((directory/'foundation/foundation.json').read_text())['passed'],'reference foundation gate must pass before neural checks')
    n=calibrate(5,64);(directory/'normalizers.json').write_text(json.dumps(n.asdict(),indent=2));rows={}
    cases=tuple(physical_case('train',p,0,h=.16,dt=.02) for p in PDES);b=batch(cases,n,sensors=8,queries=8,device=device)
    weak=WeakConfig(2,2,4,256)
    for name in ('A_GEO','C_GEO'):
        model=create(name,n,'medium',seed=903,device=device);count=model.parameter_count;require(count>=5_000_000,'actual full model below 5M')
        sync=lambda: torch.cuda.synchronize() if device=='cuda' else torch.mps.synchronize() if device=='mps' else None
        sync();t=time.perf_counter();report=Adapter(name,weak).loss(model,b,detailed=True);report.total.backward();sync();seconds=time.perf_counter()-t
        gradients={key:float(sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5) for key,module in model.named_children()}
        require(all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()),'nonfinite production gradients')
        require(all(v>0 for k,v in gradients.items() if k!='coordinates'),'unreached production parameter module')
        row={'medium_parameters':count,'inventory':model.inventory(),'device':device,'dtype':'float32','mixed_3PDE_adapter_forward_backward_seconds':seconds,'production_module_gradient_norms':gradients,
             'loss':{k:float(v.detach()) for k,v in report.unweighted.items()},'smoke_discretization':{'reference_h':.16,'reference_dt':.02,'queries':8,'sensors':8,'weak':asdict(weak)},
             'warning':'Coarse reference and reduced quadrature are smoke checks, not reference fidelity or trained accuracy.'}
        model.zero_grad(set_to_none=True)
        # Real full320-token path, separate from reduced-mesh weak smoke.
        fine=physical_case('validation','heat',0,h=.04);fb=batch([fine],n,sensors=0,queries=5,device=device);contexts=model.prepare_context(fb.specs,fb.observations)
        require(contexts[0].memory.shape==(320,model.config.width),'default full 64/256 memory not exercised')
        out=model.predict(contexts,fb.queries);out.square().mean().backward();row['full_memory_shape']=list(contexts[0].memory.shape);row['full_memory_output_shape']=list(out.shape)
        del contexts,out,fb,fine;model.zero_grad(set_to_none=True)
        large=create(name,n,'large',seed=902,device='cpu');row['large_parameters']=large.parameter_count;row['large_inventory']=large.inventory();del large;gc.collect()
        if updates:
            # Actual established loop, fixed cases, unchanged schedule. 1+1+1
            # optimizer calls including the replay, less than the five-step cap.
            config=Config(architecture=name,capacity='medium',device=device,preset='smoke',updates=2,query_count=8,microbatch=1,accumulation=3,effective_batch=3,
                          validation_query_count=8,validation_cases_per_pde=1,reference_h=.16,reference_dt=.02,sensors=8,spatial_order=2,time_order=2,weak_tests=4,physics_chunk=256,run_tag='phaseG-bounded-'+device,root_seed=903)
            engine=Engine(config,n,directory/'bounded_runs',model=model)
            first=engine.update(b);saved=engine.save_checkpoint();second=engine.update(b);expected=state_identity(engine.model)
            reloaded=resume(saved,directory/'resume_replay',device=device);replayed=reloaded.update(b)
            discrepancy=max(float((engine.model.state_dict()[k]-reloaded.model.state_dict()[k]).abs().max()) for k in engine.model.state_dict())
            require(expected==state_identity(reloaded.model) and discrepancy==0.,'resume next update differs from uninterrupted execution')
            row['bounded_updates']={'actual_optimizer_calls':3,'successful_steps_original':engine.state.successful,'checkpoint_at_step':1,'resume_replay':2,'max_parameter_discrepancy':discrepancy,'losses':[first.total,second.total],
                                    'first_gradient':first.gradient_before_clip,'second_gradient':second.gradient_before_clip,'local_checkpoint':str(saved),'local_Drive_states':engine.state.persistence,
                                    'sampler_restored':reloaded.state.sampler_cursor==engine.state.sampler_cursor,'scheduler_restored':reloaded.scheduler.state_dict()==engine.scheduler.state_dict()}
            # End validation uses independent validation fields, no updates.
            validation=engine.validate();row['validation']=validation.metrics
            del engine,reloaded
        rows[name]=row;del model,report;gc.collect();print(name,'actual medium forward/backward and bounded resume checks passed',flush=True)
    result={'schema':'phase-G-verification-v1','device':device,'environment':environment_report(),'architectures':rows,'training':'bounded smoke only; large untrained',
            'seconds':time.perf_counter()-start,'CUDA_Drive':'unverified unless run on actual Colab with mounted Drive','legacy_scientific_identity':scientific_identity()}
    path=directory/f'models-{device}.json';path.write_text(json.dumps(result,indent=2));return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--device',choices=['cpu','mps','cuda'],default='cpu');p.add_argument('--bounded-updates',action='store_true');args=p.parse_args()
    r=run(args.device,args.bounded_updates);print('Passed in',r['seconds'],'seconds')

"""Real medium architectures + real adapter backward; ZERO optimizer updates.

Bounded interface fixture: new physical priors, coarse P2 reference (not research
labels), .16 conditioning mesh, default32/48/64 sensors,8 query targets. Default
loss weights and spatial3/time3 weak objective are unchanged. Full320-memory
forward/backward is checked separately on the finer conditioning mesh.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1');os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import json,time,argparse,gc
import numpy as np
import torch
from pdeep.core import require
from pdeep.smooth.definition import *
from pdeep.smooth.data import Case,Normalizers,design,tokens,pack
from pdeep.smooth.reference import Reference
from pdeep.geometric.data import Batch
from pdeep.geometric.models import create
from pdeep.geometric.physics import Adapter,WeakConfig
from pdeep.artifact_tensors import state_identity
from pdeep.representation_checks import sync
from pdeep.runtime import resolve_device,environment_report

def main(device):
 resolve_device(device,'float32');torch.set_num_threads(1);calls=[]
 def forbidden(*a,**k):calls.append(1);raise AssertionError('No optimizer updates permitted')
 for cls in (torch.optim.AdamW,torch.optim.Adam,torch.optim.SGD):cls.step=forbidden
 n=Normalizers.fromdict(json.loads(Path('reports/smooth_finalization/normalizers.json').read_text()));cases=[];obs=[];qry=[]
 for p in PDES:
  c=Case('square',p,'train',0,.16);f=c.fields;r=Reference('square',.16,2);o=design(c,{'poisson':32,'heat':48,'wave':64}[p],observation=True);q=design(c,8)
  x=np.concatenate((o[0],q[0]));t=np.concatenate((o[1],q[1]));quant=np.concatenate((o[2],q[2]))
  if p=='poisson':clean=r.poisson([f['f']],x)[0][0]
  else:
   u,v,_=r.dynamic(p,[f['u0']],[f['v0']] if p=='wave' else None,x,t,.005);clean=np.where(quant==1,v[0],u[0]) if v is not None else u[0]
  obs.append(tokens(c,len(o[0]),n,observation=True,coordinates=o,clean=clean[:len(o[0])])[0]);qry.append(tokens(c,8,n,coordinates=q,clean=clean[len(o[0]):])[0]);cases.append(c)
 b=Batch(tuple(cases),pack(obs),pack(qry),n).to(device);rows={};weak=WeakConfig(3,3,16,256)
 for name in ('A_GEO','C_GEO'):
  print('START',name,device,flush=True);m=create(name,n,'medium',seed=809,device=device);before=state_identity(m);sync(torch.device(device));start=time.perf_counter();loss=Adapter(name,weak).loss(m,b,detailed=True);loss.total.backward();sync(torch.device(device));seconds=time.perf_counter()-start
  gradients={key:sum(float(p.grad.detach().square().sum()) for p in module.parameters() if p.grad is not None)**.5 for key,module in m.named_children()}
  require(all(p.grad is None or torch.isfinite(p.grad).all() for p in m.parameters()),'nonfinite model gradients');require(all(v>0 for k,v in gradients.items() if k!='coordinates'),'unreached trainable module');require(before==state_identity(m),'backward changed parameters')
  row={'parameters':m.parameter_count,'device':device,'dtype':'float32','observation_shape':list(b.observations.position.shape),'query_shape':list(b.queries.position.shape),'loss_unweighted':{k:float(v.detach()) for k,v in loss.unweighted.items()},'loss_weighted':{k:float(v.detach()) for k,v in loss.weighted.items()},'module_gradient_norms':gradients,'real_3PDE_adapter_forward_backward_seconds':seconds,'weak_configuration':vars(weak),'fixture_limit':'P2 h.16 reference, dt.005 for interface checking only; not certified research labels','parameter_change':0.}
  del loss;m.zero_grad(set_to_none=True);gc.collect()
  # Full model memory capacity, without the expensive fine-mesh weak integral.
  fine=Case('square','heat','train',0,.04);bb=Batch((fine,),b.observations.slice(1,2),b.queries.slice(1,2),n);ctx=m.prepare_context(bb.specs,bb.observations);y=m.predict(ctx,bb.queries,chunk=8);yc=m.predict(ctx,bb.queries,chunk=3)
  error=float(abs(y-yc).max());require(error<2e-5,'model chunk invariance failed');require(ctx[0].memory.shape==(320,192),'full320 memory not exercised');y.square().mean().backward();require(state_identity(m)==before,'dry-run changed model')
  row.update(full_memory_shape=list(ctx[0].memory.shape),chunk_max_discrepancy=error)
  del y,yc,ctx,bb,m;gc.collect();large=create(name,n,'large',seed=809,device='cpu');row['large_parameters']=large.parameter_count;del large;gc.collect();rows[name]=row;print('DONE',name,flush=True)
 result={'passed':True,'device':device,'architectures':rows,'optimizer_calls':len(calls),'runtime':environment_report(),'CUDA_Drive':'not verified unless executed natively on those services'}
 Path(f'reports/smooth_finalization/models-{device}.json').write_text(json.dumps(result,indent=2)+'\n');print('PASSED',device,'optimizer calls',len(calls))
if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');main(parser.parse_args().device)

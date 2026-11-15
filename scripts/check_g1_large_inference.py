"""Bounded accelerator arithmetic/inference check, no optimizer or reference solve."""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.smooth.data import Case,Normalizers
from pdeep.smooth.reference import Reference
from pdeep.unified_modes.model import create
from pdeep.unified_modes.architecture import inventory
from pdeep.unified_modes.preflight import preflight_batch
from pdeep.unified_modes.conditioning import MODES
from pdeep.unified.evaluation import predict

args=argparse.ArgumentParser();args.add_argument('--device',required=True,choices=('cpu','mps','cuda'));opt=args.parse_args()
def forbidden(*a,**k):raise AssertionError('Inference attempted optimizer/reference construction')
torch.optim.AdamW.__init__=forbidden;Reference.__init__=forbidden
torch.set_num_threads(1)
cert=resolve_certification(requested_capabilities={'certification_id':None}).require()
dataset=SimpleNamespace(norms=Normalizers.fromdict(cert['normalizers']),require_case=lambda case:None,
                        cache=SimpleNamespace(evaluate=lambda c,x,t,q:np.zeros(len(x),np.float64)))
rows=[];start=time.perf_counter()
with torch.no_grad():
    for arch in ('A_GEO','C_GEO'):
        model=create(arch,dataset.norms,'large',device=opt.device);case=Case('hole','wave','validation',0,.12)
        for mode in MODES:
            batch=preflight_batch(dataset,case,mode,16,opt.device)
            a,seconds=predict(model,batch.cases[0],batch.observations,batch.queries,chunk=16)
            b,_=predict(model,batch.cases[0],batch.observations,batch.queries,chunk=5)
            error=float(abs(a-b).max());assert error<2e-6*max(1.,float(abs(a).max()))
            rows.append({'architecture':arch,'capacity':'large','trainable_parameters':inventory(model)['trainable_parameters'],'mode':mode,'device':str(next(model.parameters()).device),'query_chunk_max_error':error,'prediction_seconds':seconds})
        del model
result={'scope':'large untrained hole/wave arithmetic inference; no reference-accuracy or T4-training claim','rows':rows,'seconds':time.perf_counter()-start,'optimizer_steps':0,'reference_solves':0}
p=Path('reports/g1_large')/(opt.device+'_inference.json');p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

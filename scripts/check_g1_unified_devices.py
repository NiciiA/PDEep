"""Bounded medium-model derivative/device control, no optimizer or PDE solves.

Zero field targets are an ARITHMETIC fixture, not numerical reference evidence.
Real C3 auxiliary fields/frozen normalizers/typed geometry and observations are used.
"""
import argparse,json,time
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from pdeep.runtime import resolve_device,environment_report
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.smooth.certification import normalizers
from pdeep.smooth.data import Case
from pdeep.geometric.models import create
from pdeep.unified.config import DatasetConfig,MODELS
from pdeep.unified.workflow import configurations
from pdeep.unified.data import Provider
from pdeep.unified.loss import UnifiedAdapter
from pdeep.artifact_io import atomic_json
from pdeep.representation_checks import sync


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--device',default='cpu');parser.add_argument('--output',default='reports/g1_unified/device.json');args=parser.parse_args()
    device,_=resolve_device(args.device,'float32');record=resolve_certification(requested_capabilities={'certification_id':None}).require();norms=normalizers(record)
    class Cache:
        def evaluate(self,c,x,t,q,**kw):return np.zeros(len(x),np.float64)
    dataset=SimpleNamespace(norms=norms,cache=Cache(),require_case=lambda c:None)
    provider=Provider(dataset);D=DatasetConfig(preset='smoke',stream_updates=30000,stream_queries=32);configs=configurations(D,device=str(device),query_count=32,query_chunk=17,experiment_id='zero-update-device-control')
    rows=[];start=time.perf_counter()
    for cfg in configs:
        model=create(cfg.architecture,norms,'medium',seed=42,device=str(device));adapter=UnifiedAdapter(cfg)
        for p in ('poisson','heat','wave'):
            c=Case('annulus',p,'train',2);batch,_=provider.batch((c,),count=32,training=False,device=str(device));model.zero_grad(set_to_none=True)
            sync(device);begin=time.perf_counter();loss=adapter.loss(model,batch,detailed=True);loss.total.backward();sync(device)
            grads=[v.grad for v in model.parameters() if v.grad is not None];assert grads and all(torch.isfinite(v).all() for v in grads)
            assert {v.device.type for v in grads}=={device.type}
            assert all(v.device==next(model.parameters()).device for v in grads)
            model.zero_grad(set_to_none=True)
            with torch.no_grad():
                ctx=model.prepare_context(batch.specs,batch.observations);a=model.predict(ctx,batch.queries,chunk=32);b=model.predict(ctx,batch.queries,chunk=11)
            err=float((a-b).abs().max());assert err<5e-5
            rows.append({'model':cfg.model_key,'pde':p,'geometry':'annulus','capacity':'medium','parameters':sum(v.numel() for v in model.parameters()),
                'device':str(device),'dtype':str(a.dtype),'loss_terms':list(loss.weighted),'gradient_paths':adapter.last['case_gradient_norms'],
                'max_query_chunk_difference':err,'seconds':time.perf_counter()-begin,'query_shape':list(a.shape),'kinematic_queries':512 if p=='wave' and cfg.architecture=='A_GEO' else 0})
            adapter.cache.clear_queries();del loss,ctx,a,b,batch,grads
        del model,adapter
        if device.type=='mps':torch.mps.empty_cache()
    out={'scope':'actual medium neural device/derivative arithmetic control; zero targets, not field accuracy or reference qualification',
         'environment':environment_report(),'device':str(device),'rows':rows,'seconds':time.perf_counter()-start,'optimizer_steps':0,'CUDA_verified':device.type=='cuda','Drive_verified':False}
    atomic_json(args.output,out);print(json.dumps({'device':str(device),'controls':len(rows),'seconds':out['seconds'],'max_chunk_difference':max(r['max_query_chunk_difference'] for r in rows),'optimizer_steps':0},indent=2))

if __name__=='__main__':main()

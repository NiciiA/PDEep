"""Explicit bounded preflight on existing labels. Never prepares data or trains.

--full runs the unchanged memory/gradient suite through the feasibility adapter.
Default runs one real four-case accumulated backward per selected large model.
CUDA measurements are reported only when actually run on CUDA.
"""
import argparse,json,resource,time,sys
from pathlib import Path
import torch
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.preparation.loading import load_dataset
from pdeep.preparation.feasibility_training import configurations
from pdeep.preparation.preflight import four_case_control,run_preflight
from pdeep.smooth.reference import Reference
from pdeep.smooth.data import Case

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--manifest',required=True);parser.add_argument('--model',choices=('all','A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001'),default='all');parser.add_argument('--device',choices=('cpu','cuda'),default='cpu');parser.add_argument('--full',action='store_true');args=parser.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():parser.error('CUDA requested but unavailable; no CPU fallback')
    def forbidden(*a,**kw):raise AssertionError('reference solve or optimizer forbidden during preflight')
    Reference.__init__=forbidden;Case.evaluate=forbidden;torch.optim.AdamW.__init__=forbidden;torch.set_num_threads(2)
    d=load_dataset(args.manifest,resolve_certification(requested_capabilities={'certification_id':None}).require())
    keys=('A_GEO/KIN_001','C_GEO/FIELD_ONLY','C_GEO/ALL_001') if args.model=='all' else (args.model,)
    configs=configurations(d.config,models=keys,device=args.device,query_chunk=2048,experiment_id='bounded-preflight-no-training')
    out=Path('reports/g1_preflight_loading');out.mkdir(parents=True,exist_ok=True)
    if args.full:
        path,result=run_preflight(configs,d,out);(out/'full-preflight.json').write_text(json.dumps({'path':path,'result':result},indent=2)+'\n');return
    for c in configs:
        start=time.perf_counter();r=four_case_control(c,d);r.update(total_wall_seconds=time.perf_counter()-start,host_peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if sys.platform=='darwin' else 1024))
        (out/(c.model_key.replace('/','_')+'.json')).write_text(json.dumps(r,indent=2)+'\n')
        print(json.dumps({k:r[k] for k in ('model','status','step_wall_seconds','host_peak_RSS_bytes','peak_allocated_bytes','peak_reserved_bytes','effective_cases','supervised_queries','optimizer_steps')}),flush=True)

if __name__=='__main__':main()

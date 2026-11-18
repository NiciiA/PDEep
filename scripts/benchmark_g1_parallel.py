"""Bounded CPU process scaling; production P4 only with explicit --production."""
import argparse,json,tempfile,time
from pathlib import Path
import numpy as np
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.unified.config import DatasetConfig,case_index
from pdeep.preparation.parallel import prepare_dataset
from pdeep.preparation.resources import budget
from pdeep.artifact_io import read_payload

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--production',action='store_true');parser.add_argument('--workers',default='1,2,4,0');args=parser.parse_args()
    cert=resolve_certification(requested_capabilities={'certification_id':None}).require()
    families=('square','hole') if args.production else ('square','rectangle','lshape','disk','ellipse','hole','annulus','polygon')
    config=DatasetConfig(preset='smoke',pdes=('poisson','heat','wave'),families=families,counts=(4,2,2),stream_updates=12,stream_queries=64,stream_validation_queries=8,stream_grid=4,qc_points=8)
    selected=[(p,g,'train',case_index(g,1)) for g in families for p in config.pdes];baseline={};rows=[]
    for requested in map(int,args.workers.split(',')):
        control=None if args.production else {'h':.12,'degree':2}
        fake=json.loads(json.dumps(cert))
        if control:
            for g in families:fake['mesh_records'][g]['production']['free_DOFs']=2000
        try:policy=budget(fake,families,requested=requested,block_cases=1,needs_finer=False)
        except ValueError as error:rows.append({'requested_workers':requested,'status':'skipped_unsafe','reason':str(error)});continue
        if any(r.get('actual_workers')==policy['workers'] for r in rows):rows.append({'requested_workers':requested,'status':'same_as_already_measured','actual_workers':policy['workers']});continue
        with tempfile.TemporaryDirectory(prefix='g1-process-benchmark-') as tmp:
            start=time.perf_counter();data=prepare_dataset(config,tmp,cert,workers=requested,block_cases=1,_selected_keys=selected,_control=control,progress=lambda *a,**k:None)
            seconds=time.perf_counter()-start;errors=[]
            for cid,rel in data.index['records'].items():
                _,p=read_payload(data.root/rel);labels=p['labels']
                if cid in baseline:
                    assert set(labels)==set(baseline[cid]);error=max(float((labels[k]-baseline[cid][k]).abs().max()) for k in labels);errors.append(error);assert error<1e-9
                else:baseline[cid]=labels
            assert len(data.index['records'])==len(selected),(data.index['quarantine'],data.index['preparation_profile'].get('execution_failures'))
            profile=data.index['preparation_profile'];row={'requested_workers':requested,'actual_workers':policy['workers'],'status':'passed','cases':len(selected),'wall_seconds':seconds,'speedup_vs_one':rows[0]['wall_seconds']/seconds if rows and rows[0].get('wall_seconds') else 1.,'peak_sampled_aggregate_RSS_bytes':profile['peak_aggregate_RSS_bytes'],'worker_peak_RSS_bytes':{g:r.get('peak_self_RSS_bytes') for g,r in profile['worker_profiles'].items()},'maximum_label_discrepancy':max(errors,default=0.),'numerical_QC_passed':True,'resource_policy':policy,'scope':'production P4/.02/.002 per-case controls' if args.production else 'NONPRODUCTION P2/.12 process mechanics/scaling control; not reference accuracy certification'}
            row['per_case_profiles']=profile['case_metrics'];row['per_task_wall_seconds']=profile['task_case_seconds'];row['per_task_CPU_seconds']=profile.get('task_CPU_seconds',{});row['operator_profiles']={g:r.get('operator_profile',{}) for g,r in profile['worker_profiles'].items()}
            rows.append(row);print(json.dumps(row),flush=True)
    out=Path('reports/g1_feasibility');out.mkdir(parents=True,exist_ok=True);path=out/('parallel-P4.json' if args.production else 'parallel-P2-control.json');path.write_text(json.dumps({'rows':rows,'training_updates':0,'full_dataset_generated':False},indent=2)+'\n')

if __name__=='__main__':main()

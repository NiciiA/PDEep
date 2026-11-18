"""Bounded original/optimized CPU64 comparisons; never a research dataset run."""
import argparse,json,time,cProfile,pstats,tempfile
from dataclasses import replace
from pathlib import Path
import numpy as np
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.smooth.data import Case
from pdeep.smooth.certification import normalizers
from pdeep.smooth.reference import Reference
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.unified.config import DatasetConfig
from pdeep.unified.dataset import quality_check_full,inventory
from pdeep.unified.streams import prepare_labels as original_labels,encounters as original_plan
from pdeep.preparation.numerics import FastReference,PoissonBasis,Sampler
from pdeep.preparation.labels import prepare_labels,Journal
from pdeep.preparation.planning import encounters
from pdeep.preparation.engine import References
from pdeep.artifact_io import write_artifact

def run():
    p=argparse.ArgumentParser();p.add_argument('--output',default='reports/g1_preparation');p.add_argument('--dynamic-encounters',type=int,default=12);args=p.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True);cert=resolve_certification(requested_capabilities={'certification_id':None}).require();norms=normalizers(cert)
    config=DatasetConfig(preset='smoke',stream_updates=3,stream_queries=4096,stream_grid=4,qc_points=8);rows=[]
    def measure(fn):
        start=time.perf_counter();value=fn();return value,time.perf_counter()-start
    with tempfile.TemporaryDirectory(prefix='g1-preparation-bounded-') as folder:
        phases={};pool=References(folder,phases);r=pool.get('square');basis,basis_seconds=measure(lambda:PoissonBasis(r,Case('square','poisson','train',4).fields['f']))
        for pde in ('poisson','heat','wave'):
            c=Case('square',pde,'train',4);n=121 if pde=='poisson' else args.dynamic_encounters;events=[(i,0,64) for i in range(n)]
            cold,cold_seconds=measure(lambda:FixedReference(c,reference_object=r));del cold
            old,old_s=measure(lambda:FixedReference(c,reference_object=r));new,new_s=measure(lambda:FastReference(c,reference_object=r,poisson_basis=basis if pde=='poisson' else None))
            a,a_s=measure(lambda:original_labels(c,old,config,events));profiles={};(b,complete),b_s=measure(lambda:prepare_labels(c,new,config,events,profile=profiles))
            assert complete and set(a)==set(b)
            maximum=max(float((a[k]-b[k]).abs().max()) for k in a);rms=max(float(np.sqrt(np.mean((a[k].numpy()-b[k].numpy())**2))) for k in a)
            # Compare quantities independently at fixed arbitrary times, t0/early/t1.
            x=c.problem.mesh.sample(256,np.random.default_rng(71));t=np.resize([0.,.00013,.0073,.02,.137,.731,1.],len(x));q=np.arange(len(x))%2 if pde=='wave' else np.zeros(len(x),int)
            ya=old.evaluate(x,t,q);yb=new.evaluate(x,t,q);errors=[]
            for k in sorted(set(q)):
                e=(yb[q==k]-ya[q==k])/norms.scale(pde,int(k));errors.append({'quantity':'v' if k else 'u','RMS_standardized':float(np.sqrt(np.mean(e*e))),'maximum_standardized':float(abs(e).max())})
            assert max(v['maximum_standardized'] for v in errors)<1e-8
            qc,qc_s=measure(lambda:quality_check_full(c,new,norms,config,finer=False));assert qc['passed'],qc
            journal=Journal(folder,c,config);(cached,done),journal_s=measure(lambda:prepare_labels(c,new,config,events,journal=journal));assert done and set(cached)==set(a)
            path,serialization_s=measure(lambda:write_artifact(Path(folder)/pde,'bounded-case',{'artifact_type':'diagnostic_bundle','architecture_id':'REFERENCE_ONLY','run_id':'bounded-preparation-benchmark'},{'labels':b},required=('labels',)))
            rows.append({'pde':pde,'geometry':'square','encounters':n,'unique_designs':len(a),'points':sum(v.numel() for v in a.values()),'free_DOFs':len(r.space.free),'transient_trajectory_bytes':old.bytes,'cold_factor_or_solve_seconds':cold_seconds,'old_warm_case_solve_seconds':old_s,'optimized_case_solve_seconds':new_s,'old_labels_seconds':a_s,'optimized_labels_seconds':b_s,'label_speedup':a_s/b_s,'optimized_labels_with_journal_seconds':journal_s,'QC_seconds':qc_s,'QC_passed':True,'serialization_verified_readback_seconds':serialization_s,'maximum_physical_label_error':maximum,'maximum_design_RMSE_physical':rms,'arbitrary_time_errors':errors,'evaluation_profile':profiles,'warm_compute_speedup_including_unchanged_QC':(old_s+a_s+qc_s)/(new_s+b_s+qc_s),'scope':'same exact designs and certified production P4 states; no Drive timing; no research training'})
            print(json.dumps(rows[-1]),flush=True);del old,new,a,b,cached
        small=replace(config,stream_updates=1000,stream_queries=16);inv=inventory(small);a,old_plan_s=measure(lambda:original_plan(small,inv));b,new_plan_s=measure(lambda:encounters(small,inv));assert a==b
        report={'scope':'bounded production P4 CPU64 controls, square;121 Poisson/12 dynamic encounters by default; not entire research dataset','phase_setup_seconds':phases,'poisson_basis_one_time_seconds':basis_seconds,'poisson_basis_bytes':basis.bytes,'rows':rows,'planning_1000_updates':{'old_seconds':old_plan_s,'optimized_seconds':new_plan_s,'speedup':old_plan_s/new_plan_s,'exact_events_equal':True},'actual_Drive_tested':False,'research_dataset_generated':False,'neural_updates':0}
        (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':run()

"""One production unit response, bounded runtime; no dataset qualification/training."""
import argparse,json,resource,signal,time
from tempfile import TemporaryDirectory
from pathlib import Path
import numpy as np
from pdeep.smooth.data import Case
from pdeep.unified.reference import LinearCache,CACHE_SCHEMA
import pdeep.unified.reference as module
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.artifact_io import atomic_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--pde',choices=('poisson','heat','wave'),default='wave');p.add_argument('--family',default='square');p.add_argument('--max-seconds',type=int,default=180);p.add_argument('--output',default='reports/g1_unified/production_wave_compact_control.json');args=p.parse_args()
    record=resolve_certification(requested_capabilities={'certification_id':None}).require();case=Case(args.family,args.pde,'train',0);captured={};start=time.perf_counter()
    original=module.FixedReference
    def capture(c):
        full=FixedReference(c);captured['full']=full;return full
    module.FixedReference=capture
    def stop(*a):raise TimeoutError('bounded production unit-response control limit')
    old_handler=signal.signal(signal.SIGALRM,stop);signal.alarm(args.max_seconds)
    try:
        with TemporaryDirectory(prefix='g1-reference-control-') as tmp:
            cache=LinearCache(tmp,record);cache.prepare(case,max_new_responses=1);role=next(iter(case.fields));m,payload=cache._get(case,role,0);full=captured['full']
            # Same arbitrary physical coordinates and non-grid times for full/compressed output.
            x=case.problem.mesh.sample(31,np.random.default_rng(7193));t=np.r_[0.,.0073,.02,.237,1.,np.random.default_rng(7194).uniform(0,1,26)];q=np.arange(len(x))%2 if args.pde=='wave' else np.zeros(len(x),int)
            S=full.reference.space.sample_operator(x);Y=S@payload['spatial'].numpy();T=payload['temporal'].numpy();n=m['snapshots'];h=m['dt']
            if args.pde=='poisson':value=Y@T[:,0]
            else:
                k=np.minimum((t/h).astype(int),n-2);z=(t-k*h)/h;sample=lambda cols:np.einsum('ij,ji->i',Y,T[:,cols]);ul,ur=sample(k),sample(k+1);vl,vr=sample(n+k)/h,sample(n+k+1)/h
                u=(2*z**3-3*z*z+1)*ul+(z**3-2*z*z+z)*h*vl+(-2*z**3+3*z*z)*ur+(z**3-z*z)*h*vr
                v=(6*z*z-6*z)/h*ul+(3*z*z-4*z+1)*vl+(-6*z*z+6*z)/h*ur+(3*z*z-2*z)*vr;value=np.where(q==1,v,u)
            truth=full.evaluate(x,t,q);errors={'v' if k else 'u':float(np.max(abs(value[q==k]-truth[q==k]))) for k in np.unique(q)}
            assert all(errors[k]<=m['compression']['error_bound'][k]+1e-10 for k in errors)
            out={'status':'passed','cache_schema':CACHE_SCHEMA,'source':cache.execution_source,'maximum_errors':errors,'compression':m['compression'],
                 'stored_tensor_bytes':m['stored_tensor_bytes'],'full_snapshot_bytes':m['full_snapshot_bytes'],'propagation_seconds':m['preparation_seconds'],
                 'sampled_queries':len(x),'free_DOFs':len(full.reference.space.free)}
    except TimeoutError as error:out={'status':'bounded_timeout','error':str(error),'full_reference_completed':bool(captured)}
    finally:signal.alarm(0);signal.signal(signal.SIGALRM,old_handler);module.FixedReference=original
    out.update(scope='ONE production-policy P4 unit response; not expanded physical-case qualification',pde=args.pde,family=args.family,
               seconds=time.perf_counter()-start,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    atomic_json(args.output,out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()

"""Bounded actual P4 block-vs-single evolution, no dataset run or training."""
import argparse,json,time,tempfile
from pathlib import Path
import numpy as np
from pdeep.smooth.data import Case
from pdeep.smooth.certification import normalizers
from pdeep.learning_diagnostics.certificates import resolve_certification
from pdeep.learning_diagnostics.reference_cache import FixedReference
from pdeep.preparation.evolution import Evolution
from pdeep.preparation.engine import References
from pdeep.unified.dataset import quality_check_full
from pdeep.unified.config import DatasetConfig

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--block',type=int,default=2);parser.add_argument('--family',default='square');args=parser.parse_args()
    cert=resolve_certification(requested_capabilities={'certification_id':None}).require();norms=normalizers(cert);rows=[];config=DatasetConfig(preset='smoke',qc_points=8)
    with tempfile.TemporaryDirectory(prefix='g1-block-control-') as tmp:
        phases={};r=References(tmp,phases).get(args.family)
        for pde in ('heat','wave'):
            cases=[Case(args.family,pde,'train',4*i) for i in range(args.block)]
            op=Evolution(r,pde);expected=[];walls=[];cpus=[];x=cases[0].problem.mesh.sample(128,np.random.default_rng(71));t=np.resize([0.,.00013,.0073,.02,.137,.731,1.],len(x));q=np.arange(len(x))%2 if pde=='wave' else np.zeros(len(x),int)
            for c in cases:
                wall=time.perf_counter();cpu=time.process_time();old=FixedReference(c,reference_object=r);walls.append(time.perf_counter()-wall);cpus.append(time.process_time()-cpu)
                expected.append(old.evaluate(x,t,q));del old
            block=op.propagate(cases,memory_bytes=3_000_000_000);errors=[];qc=[]
            for c,full,truth in zip(cases,block,expected):
                y=full.evaluate(x,t,q);errors.append({str(k):{'max_standardized':float(np.max(abs(y[q==k]-truth[q==k]))/norms.scale(pde,int(k))),'RMS_standardized':float(np.sqrt(np.mean((y[q==k]-truth[q==k])**2))/norms.scale(pde,int(k)))} for k in set(q)})
                qc.append(quality_check_full(c,full,norms,config,finer=False));assert qc[-1]['passed'],qc[-1]
            assert max(v['max_standardized'] for e in errors for v in e.values())<1e-8
            timing=block[0].block_timing;rows.append({'pde':pde,'geometry':args.family,'cases':args.block,'old_wall_seconds':sum(walls),'old_CPU_seconds':sum(cpus),'block':timing,'wall_speedup':sum(walls)/timing['wall_seconds'],'CPU_speedup':sum(cpus)/timing['CPU_seconds'],'errors':errors,'QC_passed':all(v['passed'] for v in qc)})
            print(json.dumps(rows[-1]),flush=True);del block
    out=Path('reports/g1_feasibility');out.mkdir(parents=True,exist_ok=True);(out/f'evolution-{args.family}-B{args.block}.json').write_text(json.dumps({'scope':'bounded original production P4 propagation; no research dataset/training','assembly':phases,'rows':rows},indent=2)+'\n')

if __name__=='__main__':main()

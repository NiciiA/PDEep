"""Explicit frozen B0+Classical benchmark. Never trains or updates a checkpoint."""
from pathlib import Path
import argparse,json
from pdeep.post_training_benchmark import run_benchmark

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--diagnosis',type=Path,default=Path('reports/b0_diagnosis/diagnosis.json'))
    p.add_argument('--output',type=Path,default=Path('reports/post_training_optimization'))
    p.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    p.add_argument('--repeats',type=int,default=7)
    args=p.parse_args();r=run_benchmark(args.diagnosis,output=args.output,device=args.device,repeats=args.repeats)
    from pdeep.post_training_report import write_report
    write_report(r,args.output)
    print(json.dumps({'cases':r['cases_per_PDE'],'device':r['device'],'seconds':r['elapsed_seconds'],'neural_updates':0},indent=2))

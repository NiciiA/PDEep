"""Read-only diagnosis of a selected saved T1 evaluation; no training."""
import argparse,json
from pathlib import Path
from pdeep.b0_diagnosis import investigate

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--result',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',choices=('cpu','mps','cuda'),default='cpu')
    args=parser.parse_args()
    r=investigate(args.result,output=args.output,device=args.device)
    from pdeep.b0_report import write_report
    write_report(r,args.output)
    print(json.dumps({'cases':r['cases_per_PDE'],'MSE':r['macro_standardized_field_MSE'],'seconds':r['elapsed_seconds']},indent=2))

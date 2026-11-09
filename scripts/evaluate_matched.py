"""Stage 7B: tiny validation-only evaluation; no training or test-split launch."""
from pathlib import Path
import argparse,json
from pdeep.diagnostics import source_identity
from pdeep.evaluation import EvaluationConfig,evaluate
from pdeep.evaluation_plan import make_plan
from pdeep.evaluation_inference import available_methods
from pdeep.evaluation_plotting import create_plots,write_report,comparison_table


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--device',default='cpu',choices=['cpu','mps','cuda'])
    parser.add_argument('--chunk',type=int,default=1024)
    args=parser.parse_args();root=args.root.resolve()
    methods,norm=available_methods(root,device=args.device);config=EvaluationConfig(chunk=args.chunk)
    plan=make_plan(norm,source_id=source_identity(root)['sha256'],config_id=config.identity,case_ids=(0,))
    result=evaluate(plan,methods,norm,config=config,output=root/'reports/matched_evaluation')
    plots=create_plots(result,plan);report=write_report(result,plan,plots)
    pointer={'directory':str(Path(result['directory']).relative_to(root)),'report':str(report.relative_to(root)),
             'plan_identity':plan.identity,'scientific_identity':result['identity']}
    (root/'reports/latest_evaluation.json').write_text(json.dumps(pointer,indent=2)+'\n')
    print(comparison_table(result));print(json.dumps(pointer,indent=2))

if __name__=='__main__':main()

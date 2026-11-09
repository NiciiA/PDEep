"""Bounded Stage 3 retrieval review; does not start Stage 4 or train anything."""
import argparse
import json
from pathlib import Path

from pdeep.core import digest
from pdeep.diagnostics import source_identity
from pdeep.retrieval_checks import kernel_checks,retrieval_checks
from pdeep.runtime import environment_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",choices=("cpu","mps","cuda"),default="cpu")
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args();root=args.root.resolve()
    source=source_identity(root)
    kernels=kernel_checks()
    require_pass=kernels["passed"]
    if not require_pass:raise RuntimeError("Gaussian kernel verification failed")
    neural=retrieval_checks(device=args.device)
    legacy=json.loads((root/"docs/retrieval_topk_audit_v1.json").read_text())
    report={"version":"gaussian-retrieval-v2","passed":kernels["passed"] and neural["passed"],
            "source":source,"environment":environment_report(),"kernels":kernels,"neural":neural,"legacy":legacy,
            "scope":"Stage 3 review only; Stage 4 not started; no training"}
    run=digest({"source":source["sha256"],"device":args.device,"version":report["version"]})
    directory=root/"reports/retrieval_review"/run;directory.mkdir(parents=True,exist_ok=True)
    report["run_id"]=run
    path=directory/"retrieval_checks.json";path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"report":str(path),"device":args.device,
                      "kernel_max_derivative_error":max(c["max_abs_error"] for c in kernels["checks"] if 'analytic_derivative' in c["name"]),
                      "chunk_error":neural["query_chunk_max_error"],"individual_query_error":neural["batched_individual_max_error"],
                      "finite_difference_scaled_error":neural["finite_difference_max_scaled_error"]},indent=2))


if __name__=="__main__":main()

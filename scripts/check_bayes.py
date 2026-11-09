"""Bounded numerical acceptance check for exact T1 Gaussian D."""
from pathlib import Path
import argparse
import json

from pdeep.bayes_checks import bayesian_checks
from pdeep.core import digest
from pdeep.data import CaseDataset,calibrate
from pdeep.diagnostics import ExperimentConfig,source_identity
from pdeep.runtime import environment_report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed",type=int,default=20261108)
    parser.add_argument("--draws",type=int,default=20000)
    args=parser.parse_args()
    root=args.root.resolve()
    source=source_identity(root)
    config=ExperimentConfig(stage="bayesian_D",root_seed=args.seed)
    normalizers=calibrate(args.seed)
    dataset=CaseDataset(args.seed)
    run=digest({"source":source["sha256"],"config":config.identity,"dataset":dataset.identity,"normalizers":normalizers.identity})
    directory=root/"reports/bayes_D"/run
    directory.mkdir(parents=True,exist_ok=True)
    report=bayesian_checks(args.seed,draws=args.draws,normalizers=normalizers)
    if not report["passed"]:
        raise RuntimeError("D acceptance failed: "+str([c for c in report["checks"] if not c["passed"]]))
    report.update(source=source,environment=environment_report(),config_sha256=config.identity,
                  dataset_id=dataset.identity,normalizer_id=normalizers.identity,run_id=run)
    path=directory/"bayes_checks.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"report":str(path),
                      "max_mean_form_discrepancy":report["max_mean_form_discrepancy"],
                      "max_covariance_form_discrepancy":report["max_covariance_form_discrepancy"],
                      "elapsed_seconds":report["elapsed_seconds"]},indent=2))


if __name__=="__main__":
    main()

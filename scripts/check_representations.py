"""Bounded representation checks on an explicitly selected device; no training."""
import argparse
import json
from pathlib import Path

from pdeep.core import digest
from pdeep.diagnostics import source_identity
from pdeep.runtime import capabilities,environment_report
from pdeep.representation_checks import representation_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device",default="cpu",choices=("cpu","mps","cuda"))
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=parser.parse_args()
    root=args.root.resolve()
    source=source_identity(root)
    report=representation_checks(device=args.device)
    if not report["passed"]:
        raise RuntimeError(str([c for c in report["checks"] if not c["passed"]]))
    identity=digest({"source":source["sha256"],"vocabulary":report["vocabulary_id"],"normalizers":report["normalizer_id"],"device":args.device})
    directory=root/"reports/representations"/identity
    directory.mkdir(parents=True,exist_ok=True)
    report.update(source=source,environment=environment_report(),run_id=identity)
    path=directory/"representation_checks.json"
    path.write_text(json.dumps(report,indent=2,allow_nan=False)+"\n")
    print(json.dumps({"passed":report["passed"],"device":report["device"],"elapsed_seconds":report["elapsed_seconds"],
                      "parameters":report["parameters"],"report":str(path)},indent=2))


if __name__=="__main__":main()

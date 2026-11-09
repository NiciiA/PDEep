"""Bounded reference/data diagnostic bundle; no Bayesian or learned methods."""
from pathlib import Path
import argparse
import json
import time

from pdeep.core import digest
from pdeep.data import (CaseDataset, calibrate, observations, training_queries,
                        persist_json, reload_observations, Normalizers, SENSOR_COUNTS)
from pdeep.diagnostics import source_identity, ExperimentConfig
from pdeep.runtime import environment_report
from pdeep.reference_checks import mathematical_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--seed",type=int,default=20261108)
    args=parser.parse_args()
    root=args.root.resolve()
    started=time.perf_counter()
    dataset=CaseDataset(args.seed)
    normalizers=calibrate(args.seed)
    source=source_identity(root)
    config=ExperimentConfig(stage="reference_data",root_seed=args.seed)
    identity=digest({"source":source["sha256"],"config":config.identity,"dataset":dataset.identity})
    directory=root/"reports/reference"/identity
    npath=normalizers.save(directory/"normalizers")
    persist_json(directory/"dataset.json",dataset.summary())
    checks=mathematical_checks(args.seed)
    if not checks["passed"]:
        raise RuntimeError("reference mathematical checks failed")
    records=[]
    for pde in SENSOR_COUNTS:
        case=dataset.case("train",pde,0)
        obs=observations(dataset,case,normalizers)
        path=obs.save(directory/"observations")
        restored=reload_observations(path,dataset,Normalizers.load(npath))
        assert obs.metadata.identity==restored.metadata.identity
        queries=training_queries(dataset,case)
        records.append({"pde":pde,"case_id":case.identity,"observation_id":obs.metadata.identity,
                        "query_id":queries.identity,"sensor_count":int(obs.inputs.valid.sum()),
                        "time_counts":obs.metadata.time_counts,"query_time_counts":queries.time_counts,
                        "sigma_min":float(obs.inputs.sigma.min()),"sigma_max":float(obs.inputs.sigma.max())})
    report={"identity":identity,"source":source,"environment":environment_report(),
            "normalizers":json.loads(npath.read_text()),"dataset":dataset.summary(),
            "mathematics":checks,"example_records":records,
            "device":"cpu","dtype":"float64","elapsed_seconds":time.perf_counter()-started,
            "not_implemented":["Bayesian baseline","neural models","training","posterior inference"]}
    path=directory/"reference_checks.json"
    path.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps({"report":str(path),"passed":checks["passed"],"max_abs_error":checks["max_abs_error"],
                      "elapsed_seconds":report["elapsed_seconds"],"normalizer_id":normalizers.identity},indent=2))


if __name__=="__main__":
    main()

"""G3 numerical studies only. Never imports neural architectures or an optimizer."""
import argparse
from pathlib import Path
import pdeep.wave_fidelity.study as study
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('study',choices=['temporal','analytical','original','correlation','candidate']);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.output is not None:study.OUT=a.output.resolve()
    study.OUT.mkdir(parents=True,exist_ok=True)
    getattr(study,a.study)()

"""Explicit bounded Stage7C verification; reuse compatible evidence by default."""
import argparse
from pathlib import Path
from pdeep.persistence_checks import run_persistence_checks


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--fresh',action='store_true',help='explicit new bounded64-update CPU verification; never overwrite prior runs')
    args=parser.parse_args();r=run_persistence_checks(args.root,fresh=args.fresh)
    print('PASS',r['run_id'],'| bounded successful updates',r['actual_scientific_optimizer_updates_this_suite'],'| seconds',r['elapsed_seconds'])

if __name__=='__main__':main()

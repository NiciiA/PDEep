"""Bounded scientific prerequisite gate. No neural model construction/training."""
from pdeep.geometric.checks import run_foundation
if __name__=='__main__':
    r=run_foundation('reports/geometry_phase/foundation')
    print('Geometry/reference prerequisite gates passed; seconds:',r['seconds'])

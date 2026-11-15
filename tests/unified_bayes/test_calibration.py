import numpy as np
from pdeep.unified_bayes.gaussian import condition
from pdeep.unified_bayes.calibration import marginal_diagnostics,aggregate_coverage
from .conftest import EVIDENCE


def test_independent_case_coverage_units_and_point_masses():
    rng=np.random.default_rng(83);H=rng.normal(size=(7,4));B=rng.normal(size=(1,4));sigma=rng.uniform(.1,.4,7);post=condition(H,np.zeros(7),sigma);variance=post.predict(B)['latent_variance'];errors=[];n=3500
    for _ in range(n):
        a=rng.normal(size=4);y=H@a+sigma*rng.normal(size=7);p=condition(H,y,sigma);errors.append(float((B@a-B@p.mean).item()))
    d=marginal_diagnostics(np.array(errors),np.zeros(n),np.full(n,variance[0]));errors_by_level=[]
    for v in d['levels']:
        tol=4*np.sqrt(v['nominal']*(1-v['nominal'])/n);assert abs(v['coverage_error'])<tol;errors_by_level.append({**v,'four_SE_tolerance':tol})
    d2=marginal_diagnostics(3*np.array(errors),np.zeros(n),np.full(n,9*variance[0]))
    for a,b in zip(d['levels'],d2['levels']):assert a['empirical_coverage']==b['empirical_coverage'];np.testing.assert_allclose(b['mean_interval_width'],3*a['mean_interval_width'])
    point=marginal_diagnostics(np.array([0.,1.]),np.zeros(2),np.zeros(2));assert point['point_mass_queries']==2 and point['inconsistent_point_masses']==1 and point['Gaussian_NLPD_positive_variance'] is None
    EVIDENCE['Gaussian_simulation_coverage']={'independent_cases':n,'levels':errors_by_level,'G1_empirical_coverage':'not evaluated: requires user-run unseen test workload'}


def test_case_bootstrap_not_pixels():
    rows=[]
    for i in range(8):
        # Arbitrarily many perfectly correlated pixels do not create extra cases.
        rows.append({'pde':'heat','family':'square','case_id':str(i),'quantity':'u','sensor_count':8,'time':None,'calibration':marginal_diagnostics(np.full(500,i/3),np.zeros(500),np.ones(500))})
    a=aggregate_coverage(rows,replicates=200);assert all(r['physical_cases']==8 and r['case_bootstrap_95_CI'] is not None for r in a)
    single=aggregate_coverage(rows[:1]);assert all(r['case_bootstrap_95_CI'] is None for r in single)

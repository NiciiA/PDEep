"""Marginal Gaussian diagnostics; independent PHYSICAL CASES are inference units."""
from collections import defaultdict
import numpy as np
from scipy.special import ndtr,ndtri
from ..core import require

LEVELS=(.50,.68,.80,.90,.95)


def marginal_diagnostics(truth,mean,variance,*,levels=LEVELS,pit_seed=7193):
    y,m,v=(np.asarray(a,dtype=np.float64) for a in (truth,mean,variance))
    require(y.shape==m.shape==v.shape and y.ndim==1 and len(y)>0,'invalid marginal diagnostic dimensions')
    require(np.isfinite(y).all() and np.isfinite(m).all() and np.isfinite(v).all() and (v>=0).all(),'invalid marginal diagnostic values')
    sd=np.sqrt(v);positive=v>0;point=~positive;e=y-m
    # A zero-variance Gaussian is a point mass, not an epsilon-variance density.
    atol=64*np.finfo(float).eps*np.maximum(1.,np.maximum(abs(y),abs(m)));consistent=abs(e)<=atol
    z=e[positive]/sd[positive];pit=np.empty(len(y));pit[positive]=ndtr(z)
    rng=np.random.default_rng(pit_seed);pit[point]=np.where(consistent[point],rng.uniform(size=int(point.sum())),np.where(e[point]<0,0.,1.))
    density=.5*(np.log(2*np.pi*v[positive])+e[positive]**2/v[positive])
    rows=[]
    for level in levels:
        a=1-level;critical=float(ndtri((1+level)/2));half=critical*sd;covered=np.where(positive,abs(e)<=half,consistent)
        lower,upper=m-half,m+half
        score=2*half+2/a*np.maximum(lower-y,0)+2/a*np.maximum(y-upper,0)
        rows.append({'nominal':level,'empirical_coverage':float(covered.mean()),'coverage_error':float(covered.mean()-level),
                     'mean_interval_width':float(np.mean(2*half)),'median_interval_width':float(np.median(2*half)),
                     'mean_interval_score':float(score.mean())})
    hist,_=np.histogram(pit,bins=np.linspace(0,1,11));hist=hist/len(pit)
    return {'levels':rows,'positive_variance_queries':int(positive.sum()),'point_mass_queries':int(point.sum()),
            'inconsistent_point_masses':int((point&~consistent).sum()),'standardized_error_mean':float(z.mean()) if len(z) else None,
            'standardized_error_RMS':float(np.sqrt(np.mean(z*z))) if len(z) else None,
            'Gaussian_NLPD_positive_variance':float(density.mean()) if len(density) else None,
            'PIT_histogram':hist.tolist(),'PIT_mean':float(pit.mean()),
            'density_scope':'continuous density scores exclude deterministic marginals; point masses reported separately, no variance floor',
            'coverage_scope':'marginal latent field unless future measurement noise explicitly added; NOT simultaneous joint coverage'}


def aggregate_coverage(case_rows,*,seed=42,replicates=1000):
    """Stratified case bootstrap, equal geometry; no pixel/temporal pseudo-replicates."""
    groups=defaultdict(list)
    for r in case_rows:
        for interval in r['calibration']['levels']:
            key=(r['pde'],r['quantity'],r['sensor_count'],r.get('time'),interval['nominal'],r.get('uncertainty_target','latent field'))
            groups[key].append((r,interval))
    rng=np.random.default_rng(seed);out=[]
    for key,rows in sorted(groups.items(),key=lambda v:str(v[0])):
        # One value per case/family/target/time in each group.
        keys=[(r['family'],r['case_id']) for r,_ in rows];require(len(set(keys))==len(keys),'duplicate physical case coverage entries')
        by_geometry=defaultdict(list)
        for r,v in rows:by_geometry[r['family']].append(v)
        def macro(field):return float(np.mean([np.mean([v[field] for v in values]) for values in by_geometry.values()]))
        samples=[]
        if all(len(values)>1 for values in by_geometry.values()):
            boot=[]
            for values in by_geometry.values():
                a=np.array([v['empirical_coverage'] for v in values]);boot.append(a[rng.integers(len(a),size=(replicates,len(a)))].mean(1))
            samples=np.mean(boot,axis=0)
        out.append({'pde':key[0],'quantity':key[1],'sensor_count':key[2],'time':key[3],'nominal':key[4],'uncertainty_target':key[5],
                    'empirical_coverage':macro('empirical_coverage'),'coverage_error':macro('coverage_error'),
                    'mean_interval_width':macro('mean_interval_width'),'mean_of_case_median_widths':macro('median_interval_width'),
                    'mean_interval_score':macro('mean_interval_score'),'physical_cases':len(rows),'geometries':len(by_geometry),
                    'case_bootstrap_95_CI':np.quantile(samples,[.025,.975]).tolist() if len(samples) else None,
                    'bootstrap_unit':'independent physical cases, stratified within geometry; equal-geometry macro average',
                    'bootstrap_status':'measured' if len(samples) else 'insufficient independent cases in at least one stratum'})
    return out


def aggregate_probabilistic(case_rows):
    """Case-macro proper scores/PIT; density scores exclude point masses explicitly."""
    groups=defaultdict(list)
    for r in case_rows:groups[(r['pde'],r['quantity'],r['sensor_count'],r.get('time'),r.get('uncertainty_target','latent field'))].append(r)
    out=[]
    for key,rows in sorted(groups.items(),key=lambda x:str(x[0])):
        geometry=defaultdict(list)
        for r in rows:geometry[r['family']].append(r['calibration'])
        def macro(field):
            v=[np.mean([d[field] for d in ds if d[field] is not None]) for ds in geometry.values() if any(d[field] is not None for d in ds)]
            return float(np.mean(v)) if v else None
        pit=np.mean([np.mean([d['PIT_histogram'] for d in ds],axis=0) for ds in geometry.values()],axis=0)
        out.append({'pde':key[0],'quantity':key[1],'sensor_count':key[2],'time':key[3],'uncertainty_target':key[4],
            'Gaussian_NLPD_positive_variance':macro('Gaussian_NLPD_positive_variance'),'mean_case_standardized_error':macro('standardized_error_mean'),
            'mean_case_standardized_error_RMS':macro('standardized_error_RMS'),'PIT_mean':macro('PIT_mean'),'PIT_histogram':pit.tolist(),
            'inconsistent_point_masses':sum(r['calibration']['inconsistent_point_masses'] for r in rows),'point_mass_queries':sum(r['calibration']['point_mass_queries'] for r in rows),
            'physical_cases':len(rows),'aggregation':'equal case within geometry, then equal geometry; density excludes point masses; PIT randomization uses local diagnostic RNG'})
    return out

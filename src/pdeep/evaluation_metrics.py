"""Physical-case reductions and honest uncertainty about empirical comparisons."""
import math
import numpy as np
import torch
from .core import require


def errors(prediction, target, scale, valid=None):
    p=prediction.detach().cpu().double(); t=target.detach().cpu().double()
    require(p.shape==t.shape and math.isfinite(scale) and scale>0, 'metric shape/scale mismatch')
    e=(p-t) if valid is None else (p-t)[valid.cpu()]
    require(e.numel()>0 and bool(torch.isfinite(e).all()), 'nonempty finite valid scores required')
    mse=float(e.square().mean())
    return {'physical_MSE':mse,'physical_RMSE':math.sqrt(mse),'physical_MAE':float(e.abs().mean()),
            'standardized_MSE':mse/scale**2,'count':e.numel()}


def coefficient_errors(estimate, truth, scales):
    e=estimate.detach().cpu().double()-truth.detach().cpu().double(); s=scales.detach().cpu().double()
    require(e.shape==s.shape and e.numel() in (9,18), 'coefficient dimensions')
    freq=torch.tensor([m*m+n*n for m in range(1,4) for n in range(1,4)]*(e.numel()//9))
    groups={str(int(k)):{'physical_MSE':float(e[freq==k].square().mean()),
                         'prior_standardized_MSE':float((e[freq==k]/s[freq==k]).square().mean())} for k in freq.unique()}
    return {'physical_RMSE':float(e.square().mean().sqrt()),'prior_standardized_MSE':float((e/s).square().mean()),
            'per_mode_error':e.tolist(),'frequency_m2_plus_n2':groups,
            'blocks':{name:{'physical_RMSE':float(e[i:i+9].square().mean().sqrt()),'prior_standardized_MSE':float((e[i:i+9]/s[i:i+9]).square().mean())} for name,i in (('a',0),('b',9)) if i<len(e)}}


def aggregate(cases):
    """points -> equal quantities per case -> equal cases/PDE -> equal PDE macro."""
    require(bool(cases),'nonempty cases required'); result={}
    keys=('physical_MSE','physical_MAE','standardized_MSE')
    for pde in sorted({r['pde'] for r in cases}):
        rows=[r for r in cases if r['pde']==pde]
        quantities={q:{k:sum(r['quantities'][q][k] for r in rows)/len(rows) for k in keys} for q in rows[0]['quantities']}
        for v in quantities.values():v['physical_RMSE']=math.sqrt(v['physical_MSE'])
        family={k:sum(sum(v[k] for v in r['quantities'].values())/len(r['quantities']) for r in rows)/len(rows) for k in keys}
        family['physical_RMSE']=math.sqrt(family['physical_MSE'])
        result[pde]={'cases':len(rows),'quantities':quantities,'equal_quantity_metrics':family}
    macro={k:sum(v['equal_quantity_metrics'][k] for v in result.values())/len(result) for k in keys}
    macro['physical_RMSE']=math.sqrt(macro['physical_MSE'])
    return {'by_PDE':result,'macro':macro,'physical_unit_note':'macro physical errors mix u/v units; inspect quantity rows'}


def paired_interval(left, right, *, seed=0, draws=2000):
    """Stratified paired physical-case bootstrap, never pointwise pseudo-replication."""
    require([(r['pde'],r['case_id']) for r in left]==[(r['pde'],r['case_id']) for r in right], 'unpaired comparison')
    values={p:[] for p in sorted({r['pde'] for r in left})}
    for a,b in zip(left,right):
        def score(r):return sum(v['standardized_MSE'] for v in r['quantities'].values())/len(r['quantities'])
        values[a['pde']].append(score(a)-score(b))
    estimate=float(np.mean([np.mean(v) for v in values.values()]))
    if min(map(len,values.values()))<2:
        return {'estimate':estimate,'CI95':None,'reason':'one physical case per PDE cannot estimate sampling variability','unit':'paired physical cases'}
    rng=np.random.default_rng(seed)
    boot=np.mean([rng.choice(v,(draws,len(v)),replace=True).mean(1) for v in values.values()],axis=0)
    return {'estimate':estimate,'CI95':np.quantile(boot,[.025,.975]).tolist(),'unit':'paired physical cases stratified by PDE','draws':draws}


def coverage(mean, variance, truth, valid=None):
    m=mean.detach().cpu().double(); v=variance.detach().cpu().double(); t=truth.detach().cpu().double()
    if valid is not None:m,v,t=m[valid.cpu()],v[valid.cpu()],t[valid.cpu()]
    require(m.shape==v.shape==t.shape and m.numel()>0 and bool(torch.isfinite(v).all()) and bool((v>=0).all()), 'invalid coverage variance')
    # Boundary zero variance is an exact Dirichlet constraint; numerical sin(pi)
    # remnants are handled with a CPU64 scale-aware roundoff allowance only.
    tol=32*torch.finfo(torch.float64).eps*(1+m.abs()+t.abs())
    return {str(level):float(((m-t).abs()<=z*v.sqrt()+tol).double().mean())
            for level,z in ((.5,.6744897501960817),(.9,1.6448536269514722),(.95,1.959963984540054))}


def coverage_interval(case_coverages):
    """Independent physical cases are clusters; query locations are correlated.

    Distribution-free Hoeffding intervals for the equal-PDE average of bounded
    case coverage fractions. They are intentionally broad on the tiny smoke run.
    """
    pdes=sorted({r['pde'] for r in case_coverages}); weights=[]; values=[]
    for p in pdes:
        rows=[r for r in case_coverages if r['pde']==p]
        weights.extend([1/(len(pdes)*len(rows))]*len(rows));values.extend(r['coverage'] for r in rows)
    radius=math.sqrt(.5*sum(w*w for w in weights)*math.log(40))
    return {k:{'estimate':sum(w*v[k] for w,v in zip(weights,values)),
               'CI95':[max(0,sum(w*v[k] for w,v in zip(weights,values))-radius),min(1,sum(w*v[k] for w,v in zip(weights,values))+radius)],
               'method':'bounded independent physical-case Hoeffding; no pointwise independence assumption'} for k in values[0]}

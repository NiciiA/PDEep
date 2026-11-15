"""Fixed six-method, case-paired summaries; physical units never pooled."""
from collections import defaultdict
from itertools import combinations
from pathlib import Path
import csv
import numpy as np
from ..core import require
from ..artifact_io import atomic_json
from .evaluation import CLASSICAL,NEURAL

METHODS=CLASSICAL+NEURAL
QUANTITIES=('poisson/u','heat/u','wave/u','wave/v')


def _macro(rows,metric):
    strata=defaultdict(list)
    for r in rows:
        if r.get(metric) is not None:strata[(r['pde'],r['family'],r['quantity'])].append(r[metric])
    return float(np.mean([np.mean(v) for v in strata.values()])) if strata else None


def validate_matching(rows):
    lookup=defaultdict(dict)
    for r in rows:
        require(r['split']=='test','final comparison requires unseen TEST cases; never select checkpoints on test')
        require(r['method'] in METHODS,'unregistered method')
        key=(r['case_id'],r['sensor_count'],r['quantity']);method=r['method']
        require(method not in lookup[key],'duplicate method/case/count/quantity result')
        lookup[key][method]=r
    for group in lookup.values():
        require(len({(r['observation_identity'],r['query_identity']) for r in group.values()})==1,'unmatched observations or queries across methods')
    cohort=set(lookup)
    for method in METHODS:
        present={k for k,v in lookup.items() if method in v}
        require(not present or present==cohort,'incomplete matched cohort for '+method+'; finish evaluation or explicitly select a shared bounded cohort')
    return lookup


def summarize(rows,*,seed=42,replicates=1000):
    validate_matching(rows);available=[r for r in rows if r.get('standardized_MSE') is not None];counts=sorted({r['sensor_count'] for r in rows})
    selectors=[('overall',None)]+[('sensor_count',n) for n in counts]+[('pde',p) for p in ('poisson','heat','wave')]+[('family',g) for g in sorted({r['family'] for r in rows})]+[('task',p+'/'+g) for p,g in sorted({(r['pde'],r['family']) for r in rows})]
    tables=[];default_count=max(counts) if counts else 128
    for group,key in selectors:
        subset=available if group=='sensor_count' else [r for r in available if r['sensor_count']==default_count]
        if group=='sensor_count':subset=[r for r in subset if r['sensor_count']==key]
        elif group in ('pde','family'):subset=[r for r in subset if r[group]==key]
        elif group=='task':subset=[r for r in subset if r['pde']+'/'+r['family']==key]
        for method in METHODS:
            selected=[r for r in subset if r['method']==method];v={'group':group,'key':key,'method':method,'status':'available' if selected else 'unavailable — run matched evaluation', 'sensor_count':key if group=='sensor_count' else default_count}
            for pq in QUANTITIES:
                qrows=[r for r in selected if r['pde']+'/'+r['quantity']==pq]
                for metric in ('standardized_MSE','physical_RMSE','relative_L2'):v[pq+'/'+metric]=_macro(qrows,metric)
                v[pq+'/cases']=len(qrows)
            tables.append(v)
    # Differences use exactly paired physical cases. No pixel or frame bootstrap.
    rng=np.random.default_rng(seed);pairs=[]
    for n in counts:
        for pq in QUANTITIES:
            by=defaultdict(dict)
            for r in available:
                if r['sensor_count']==n and r['pde']+'/'+r['quantity']==pq:by[r['method']][r['case_id']]=r
            for a,b in combinations(METHODS,2):
                shared=sorted(by[a].keys()&by[b].keys());strata=defaultdict(list)
                for c in shared:strata[by[a][c]['family']].append(by[a][c]['standardized_MSE']-by[b][c]['standardized_MSE'])
                if not strata:continue
                boot=[np.asarray(v)[rng.integers(len(v),size=(replicates,len(v)))].mean(1) for v in strata.values()] if all(len(v)>1 for v in strata.values()) else []
                pairs.append({'methods':[a,b],'quantity':pq,'sensor_count':n,'paired_cases':len(shared),'mean_standardized_MSE_difference':float(np.mean([np.mean(v) for v in strata.values()])),
                    'paired_case_bootstrap_95_CI':np.quantile(np.mean(boot,axis=0),[.025,.975]).tolist() if len(boot) else None,'bootstrap_unit':'physical case; stratified by geometry; CI unavailable with singleton strata'})
    cost=[]
    for method in METHODS:
        unique={ (r['case_id'],r['sensor_count']):r for r in available if r['method']==method and r['sensor_count']==default_count }
        latency=[r['cost']['inference_seconds'] for r in unique.values() if r.get('cost',{}).get('inference_seconds') is not None]
        lookup=[r['cost'].get('operator_lookup_seconds',0) for r in unique.values()]
        interp=[r['cost'].get('preprocessing_seconds',0) for r in unique.values()]
        training={r.get('checkpoint_artifact'):r.get('training_seconds') for r in unique.values() if r.get('training_seconds') is not None}
        cost.append({'method':method,'sensor_count':default_count,'case_sensor_sets':len(unique),'median_inference_seconds':float(np.median(latency)) if latency else None,'p95_inference_seconds':float(np.quantile(latency,.95)) if latency else None,'mean_inference_seconds':float(np.mean(latency)) if latency else None,
            'mean_operator_lookup_seconds':float(np.mean(lookup)) if lookup else None,'mean_interpolation_preprocessing_seconds':float(np.mean(interp)) if interp else None,'neural_one_time_training_seconds':list(training.values()),
            'timing_scope':'single measured complete query grid per case/count, CPU64 classical or recorded neural device; no unmeasured speed claim'})
    return {'methods':list(METHODS),'primary_metric':'standardized_MSE = physical MSE / frozen TRAIN quantity RMS squared',
        'aggregation':'equal physical cases within geometry, equal geometries within each PDE/quantity; overall uses largest selected sensor count; four physical quantities stay separate',
        'overall_sensor_count':default_count,'tables':tables,'paired_intervals':pairs,'cost':cost,'rows':rows,'neural_uncertainty':'unavailable: all three neural models are deterministic'}


def save(report,root):
    root=Path(root);root.mkdir(parents=True,exist_ok=True);atomic_json(root/'comparison.json',report)
    with (root/'tables.csv').open('w',newline='') as f:
        fields=list(report['tables'][0]);w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(report['tables'])
    return str(root/'comparison.json')

"""Bounded display summaries of all saved physical cases; no new scoring."""
import numpy as np
from .core import require


def representative_cases(science,case_ids=None):
    """One fixed case/PDE, selected by ID, never by error or posterior width."""
    first=next(iter(science['cases'].values()));selected={}
    for pde in sorted({r['pde'] for r in first}):
        candidates=[r for r in first if r['pde']==pde]
        wanted=None if case_ids is None else case_ids.get(pde)
        if wanted is None:row=min(candidates,key=lambda r:r['case_id'])
        else:
            matches=[r for r in candidates if r['case_id']==wanted or int(r['case_id'].rsplit('/',1)[-1])==wanted]
            require(len(matches)==1,'representative case ID is not uniquely present in the evaluated plan')
            row=matches[0]
        require(all(any(r['case_id']==row['case_id'] and r['pde']==pde for r in rows) for rows in science['cases'].values()),'representative case missing from a method')
        selected[pde]=row
    return selected


def physics_summary_table(result):
    """Equal-case RMS of residual RMSE, plus maxima; components/units stay separate."""
    lines=['| Method | PDE | Cases | Equation component | Units | Physical residual RMSE | Normalized RMSE | Max boundary abs |',
           '|---|---|---:|---:|---|---:|---:|---:|']
    for name,all_rows in result['scientific']['cases'].items():
        for pde in sorted({r['pde'] for r in all_rows}):
            rows=[r['physics'] for r in all_rows if r['pde']==pde and r.get('physics')]
            if not rows:continue
            boundary=max((max(r.get('boundary_max_abs',{}).values(),default=0.) for r in rows),default=0.)
            for j,unit in enumerate(rows[0]['units']):
                physical=np.sqrt(np.mean([r['physical_RMSE'][j]**2 for r in rows]))
                normalized=np.sqrt(np.mean([r['normalized_RMSE'][j]**2 for r in rows]))
                lines.append(f'| {name} | {pde} | {len(rows)} | {j+1} | {unit} | {physical:.5g} | {normalized:.5g} | {boundary:.3g} |')
    return '\n'.join(lines)


def coverage_summary_table(result):
    lines=['| PDE | Cases | Exact D interval | 50% coverage | 90% coverage | 95% coverage |',
           '|---|---:|---|---:|---:|---:|']
    rows=result['scientific']['cases'].get('D',[])
    for pde in sorted({r['pde'] for r in rows}):
        group=[r['uncertainty'] for r in rows if r['pde']==pde]
        keys={'coefficients':'coefficient_coverage','held-out latent':'heldout_latent_coverage','held-out measurement':'heldout_measurement_coverage'}
        groups={label:[r[key] for r in group] for label,key in keys.items()}
        for q in group[0]['field_quantities']:
            groups['field '+q+' latent']=[r['field_quantities'][q]['latent_coverage'] for r in group]
        for label,values in groups.items():
            cells=' | '.join(f"{np.mean([r[k] for r in values]):.4f}" for k in ('0.5','0.9','0.95'))
            lines.append(f'| {pde} | {len(group)} | {label} | {cells} |')
    return '\n'.join(lines)


def refinement_summary_table(result):
    lines=['| Method | PDE | Cases | Refinement k | Mean Phi | Pooled coefficient RMSE | Coefficient RMSE to D mean |',
           '|---|---|---:|---:|---:|---:|---:|']
    for name,all_rows in result['scientific']['cases'].items():
        for pde in sorted({r['pde'] for r in all_rows if 'refinement' in r}):
            rows=[r['refinement'] for r in all_rows if r['pde']==pde and 'refinement' in r]
            for k in range(len(rows[0]['Phi_by_iteration'])):
                phi=np.mean([r['Phi_by_iteration'][k] for r in rows])
                coef=np.sqrt(np.mean([r['coefficient_vs_truth'][k]['physical_RMSE']**2 for r in rows]))
                dist=np.sqrt(np.mean([r['coefficient_vs_D_mean'][k]['physical_RMSE']**2 for r in rows]))
                lines.append(f'| {name} | {pde} | {len(rows)} | {k} | {phi:.5g} | {coef:.5g} | {dist:.5g} |')
    return '\n'.join(lines)

"""Presentation-only study progress, outside scientific/execution fingerprints.

Opt in for a FUTURE invocation via this module's run_study wrapper. Never reload
or install monitoring into an active training call. Scheduling, configurations,
source IDs, optimizer loops, callbacks and checkpoint transactions remain in the
unchanged learning_diagnostics modules. Status inspection only reads artifacts.
"""
from collections import deque
from pathlib import Path
import json
import math
import statistics
from .learning_diagnostics.dashboard import Dashboard
from .learning_diagnostics.suite import SuiteDashboard, _snapshot


def selected_variants(state):
    """Use the recorded sequence, NOT JSON object-key order (canonicalized)."""
    recipes=state.get('preview',{}).get('experiments',())
    result=[]
    for name in state['order']:
        recipe=next((r for r in recipes if r['experiment']==name),{})
        order=recipe.get('variants')
        if order is None:
            # Old suites without explicit recipes: only the original fixed forms
            # have an unambiguous fallback. Never alphabetize custom variants.
            order=('FULL','FIELD_ONLY') if name=='c_physics_ablation' else ('FULL',)
        result.extend((name,label) for label in order)
    return result


def _last_step(path):
    """Bounded tail read; ignore an incomplete concurrently written last line."""
    if not path.is_file():return None
    with path.open('rb') as stream:
        stream.seek(0,2);end=stream.tell();stream.seek(max(0,end-262144));lines=stream.read().splitlines()
    for line in reversed(lines):
        try:row=json.loads(line)
        except (ValueError,UnicodeError):continue
        if row.get('status')=='successful':return row
    return None


def inspect_study(plan_or_directory):
    """Latest verified snapshot + latest complete step log. Never prepares a run.

    'running' is inferred from saved active group + sequential order, not a
    process-liveness assertion. Logged updates can exceed checkpointed updates.
    """
    if hasattr(plan_or_directory,'suite_id'):
        directory=Path(plan_or_directory.root)/'learning_diagnostics'/'suites'/plan_or_directory.suite_id
    else:directory=Path(plan_or_directory).expanduser()
    pointer=json.loads((directory/'latest.json').read_text())
    manifest=pointer['manifest']
    if not Path(manifest).is_file():manifest=pointer.get('Drive_manifest')
    if not manifest:raise ValueError('Verified suite snapshot unavailable; mount Drive or select the actual local study directory')
    state,_=_snapshot(manifest)
    order=selected_variants(state);rows=[]
    active=state.get('active_experiment');entry=state['entries'].get(active,{})
    first=next((label for name,label in order if name==active and entry.get('runs',{}).get(label,{}).get('status')!='completed'),None)
    budget=state['preview']['settings']['updates']
    for index,(name,label) in enumerate(order,1):
        run=state['entries'][name].get('runs',{}).get(label,{})
        saved=int(run.get('last_successful_update',0));observed=int(run.get('observed_update',saved));tail=None
        checkpoint=run.get('checkpoint')
        if checkpoint:tail=_last_step(Path(checkpoint).parent.parent/'steps.jsonl')
        if tail and tail.get('run_id')==run.get('run_id'):observed=max(observed,int(tail['successful']))
        status=run.get('status','pending')
        if status!='completed' and name==active and label==first:
            if state['status']=='running':status='finalizing (inferred)' if observed>=budget else 'running (inferred)'
            elif state['status'] in {'failed','interrupted'}:status=state['status']+'; resume required'
        rows.append({'index':index,'experiment':name,'variant':label,'status':status,'successful':observed,
                     'planned':budget,'checkpointed':saved,'run_id':run.get('run_id'),'checkpoint':checkpoint,
                     'latest_log_used':bool(tail)})
    return {'suite_id':state['suite_id'],'suite_status':state['status'],'snapshot':str(manifest),
            'selected_variants':[label for _,label in order],'rows':rows,'automatic_progression':True,
            'scope':'read-only saved/logged state; running is inferred, not a liveness check; completed updates after last checkpoint are not yet resumable'}


class VariantDashboard(SuiteDashboard):
    """Same handle/callback contract; variant totals and explicitly estimated ETA."""
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.samples={};self.progress={}
    @staticmethod
    def _pilot(run):
        rows=run.get('preflight',{}).get('profiles',())
        value=rows[0].get('step_wall_seconds') if rows else None
        return value if isinstance(value,(int,float)) and math.isfinite(value) and value>0 else None
    def render(self,v):
        state=self.suite_state;order=selected_variants(state);budget=state.get('preview',{}).get('settings',{}).get('updates',0)
        active=next(((name,label) for name,label in order if v.get('run_id') is not None and state['entries'][name].get('runs',{}).get(label,{}).get('run_id')==v.get('run_id')),None)
        rate=None;run={};successful=0
        if active:
            name,label=active;run=state['entries'][name]['runs'][label];successful=int(v.get('successful',0))
            samples=self.samples.setdefault(v['run_id'],deque(maxlen=21))
            elapsed=v.get('elapsed_seconds')
            if isinstance(elapsed,(int,float)) and math.isfinite(elapsed):
                if not samples or successful>samples[-1][0]:samples.append((successful,elapsed))
                durations=[(b[1]-a[1])/(b[0]-a[0]) for a,b in zip(samples,list(samples)[1:]) if b[0]>a[0] and b[1]>a[1]]
                if durations:rate=statistics.median(durations)
                elif successful and elapsed>0:rate=elapsed/successful
        current_eta=max(0,budget-successful)*rate if rate is not None else None
        if active and successful>=budget:current_eta=0.
        whole=0.;done=0;methods=[]
        for name,label in order:
            other=state['entries'][name].get('runs',{}).get(label,{})
            n=budget if other.get('status')=='completed' else int(other.get('observed_update',other.get('last_successful_update',0)))
            if (name,label)==active:n=successful
            n=min(budget,n);done+=n;remaining=max(0,budget-n)
            if not remaining:continue
            if rate is None:whole=None;continue
            own=self._pilot(run);future=self._pilot(other)
            if (name,label)==active:estimate=rate
            elif own and future:estimate=rate*future/own;methods.append('pending variants scaled by measured preflight cost')
            else:estimate=rate;methods.append('unmeasured variants projected at current variant rate')
            if whole is not None:whole+=remaining*estimate
        if not active and any(state['entries'][name].get('runs',{}).get(label,{}).get('status')!='completed' for name,label in order):whole=None
        index=order.index(active)+1 if active else None
        label=f'Variant {index}/{len(order)}: {active[1]}' if active else f'Study complete: {len(order)}/{len(order)} variants' if state.get('status')=='completed' else f'Preparing study: {len(order)} variants'
        self.progress={'label':label,'variant_index':index,'variant_total':len(order),'current_variant':active[1] if active else None,
            'current_variant_updates':successful if active else None,'current_variant_planned_updates':budget if active else None,
            'current_variant_ETA_seconds':current_eta,'whole_study_ETA_seconds':whole,
            'whole_study_successful_updates':done,'whole_study_planned_updates':budget*len(order),
            'ETA_scope':'estimated optimizer/validation/checkpoint time; excludes final diagnostic plots; '+('; '.join(sorted(set(methods))) or 'calibrating or measured current variant'),
            'selected_variants':[item[1] for item in order]}
        # Stale last-run events must not turn a completed study back into 1/5.
        if state.get('status')=='completed':
            self.progress.update(label=f'Study complete: {len(order)}/{len(order)} variants',whole_study_ETA_seconds=0.)
        self.latest.update(study_progress=self.progress)
        view={**v,'experiment':self.progress['label'],'ETA_seconds':current_eta}
        view['performance']={**v.get('performance',{}),'study_progress':self.progress}
        Dashboard.render(self,view)


def run_study(plan,*,run_mode='preview',notebook=True):
    """Opt-in presentation wrapper for a FUTURE synchronous study invocation.

    No scientific files or source identities change. The existing runner accepts
    a monitor already; this scoped adapter supplies it to run_study's internal
    call, then restores that binding. Do not call this wrapper concurrently with
    another study, or hot-reload it into a currently executing old call.
    """
    from .learning_diagnostics import optimization_studies as studies
    if run_mode=='preview':return studies.run_study(plan,run_mode=run_mode,notebook=notebook)
    runner=studies.run_diagnostic_suite
    if getattr(runner,'_variant_monitoring_wrapper',False):raise RuntimeError('A monitored study is already executing; do not start another worker')
    def monitored(plan,**kwargs):
        if kwargs.get('monitor') is None:kwargs['monitor']=VariantDashboard(suite_state={},notebook=notebook)
        return runner(plan,**kwargs)
    monitored._variant_monitoring_wrapper=True
    studies.run_diagnostic_suite=monitored
    try:return studies.run_study(plan,run_mode=run_mode,notebook=notebook)
    finally:studies.run_diagnostic_suite=runner

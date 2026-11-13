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


class WaveDashboard(VariantDashboard):
    """Presentation over two existing suites; no scheduler or persisted manager."""
    def __init__(self,plans,**kwargs):
        self.architecture=plans[0].model
        import threading,time
        self._lock=threading.RLock();self._stop=threading.Event();self._heartbeat=None
        self._started=time.monotonic();self._last_text=None;self._view={};self.phase_state={};self.failed_stage=None;self.last_successful=0;self._last_forced={}
        self._heartbeat_seconds=float(kwargs.pop('heartbeat_seconds',5.));
        self._external_renderer=kwargs.pop('renderer',None);self._fallback=False
        self._notebook=kwargs.get('notebook',True);self._plans=plans
        self.query_count=plans[0].settings.query_count;self.query_chunk=plans[0].settings.query_chunk
        self.device=plans[0].settings.device;self.device_name=self.device
        self.lookup='/'.join(sorted({getattr(p,'wave_lookup','none') for p in plans}))
        self.wave_state={'order':[p.model for p in plans],'status':'running','entries':{},'preview':{'settings':{'updates':plans[0].settings.updates},'experiments':[]}}
        for p in plans:
            entry=next(iter(p.snapshot.get('entries',{}).values()),{}) if p.snapshot else {}
            runs=entry.get('runs',getattr(p,'reused_runs',{}))
            self.wave_state['entries'][p.model]={'status':entry.get('status','pending'),'runs':{p.model+'/'+v:run for v,run in runs.items()}}
            self.wave_state['preview']['experiments'].append({'experiment':p.model,'variants':[p.model+'/'+v for v in p.settings.variant_settings]})
        super().__init__(suite_state={},renderer=self._display,**kwargs)
    def _copy_entry(self,architecture,state):
        entry=next(iter(state.get('entries',{}).values()),{'status':'pending','runs':{}})
        self.wave_state['entries'][architecture]={**entry,'runs':{architecture+'/'+label:run for label,run in entry.get('runs',{}).items()}}
    def render(self,v):
        self._view=dict(v);self.last_successful=int(v.get('successful',self.last_successful))
        if v.get('architecture'):self.architecture=v['architecture']
        if v.get('phase') is not None:self.phase_state.update(stage={'ready':'TRAINING','complete':'ARM COMPLETE'}.get(v['phase'],v['phase'].upper()),since=__import__('time').monotonic())
        if v.get('phase')=='failed' and v.get('last_error'):self.failed_stage=str(v['last_error']).split(':',1)[0]
        label=next((label for entry in self.suite_state.get('entries',{}).values() for label,run in entry.get('runs',{}).items() if v.get('run_id') and run.get('run_id')==v['run_id']),None)
        if label:self.phase_state['variant']=self.architecture+'/'+label
        local=self.suite_state
        self._copy_entry(self.architecture,local)
        try:
            self.suite_state=self.wave_state
            super().render(v)
        finally:self.suite_state=local
    def finish_architecture(self,architecture,result):
        self._copy_entry(architecture,result)
        if all(e['status']=='completed' for e in self.wave_state['entries'].values()):
            self.wave_state['status']='completed';self.render(self.latest)


    def update(self,config,state,step=None,validation=None):
        before=self.render_count
        result=super().update(config,state,step,validation)
        if state.successful-self._last_forced.get(state.run_id,-10)>=10:
            if self.render_count==before:self.render(result)
            self._last_forced[state.run_id]=state.successful
        return result

    def _display(self,text,previous):
        with self._lock:
            self._last_text=text
            if self._external_renderer is not None:
                try:return self._external_renderer(text,previous)
                except Exception as error:
                    self._fallback=True
                    if not self.errors:self.errors.append(str(error))
            return self._emit(previous)

    def _emit(self,previous=None):
        import html,time
        with self._lock:
            v=dict(self._view);p=dict(self.progress);state=dict(self.phase_state)
            def fmt(value):return '—' if value is None else f'{value:.5g}' if isinstance(value,float) else str(value)
            perf=v.get('performance',{});memory=perf.get('memory',{}) or v.get('memory',{})
            allocated=memory.get('allocated_bytes',memory.get('device_current_allocated_bytes'))
            raw=v.get('unweighted',{});persistence=v.get('persistence',{});sync=persistence.get('last_sync',{})
            completed=sum(run.get('status')=='completed' for entry in self.wave_state['entries'].values() for run in entry['runs'].values())
            total=sum(len(r['variants']) for r in self.wave_state['preview']['experiments'])
            current=state.get('variant') or p.get('current_variant') or self.architecture
            since=state.get('since',self._started);elapsed=time.monotonic()-self._started
            eta=p.get('current_variant_ETA_seconds')
            matching=next((run for entry in self.wave_state['entries'].values() for run in entry['runs'].values() if run.get('run_id')==v.get('run_id')),{})
            origin=matching.get('result_origin') or state.get('origin','fresh' if self.lookup=='none' else 'explicit resume / fresh unlisted arm')
            fraction=perf.get('clipped_fraction')
            fraction='—' if fraction is None else f'{100*fraction:.2f}%'
            whole=p.get('whole_study_ETA_seconds') if 'unmeasured' not in p.get('ETA_scope','unmeasured') else None
            step_seconds=perf.get('step_wall_seconds')
            ups=1/step_seconds if step_seconds and step_seconds>0 else v.get('updates_per_second')
            text=(f"PHASE G — WAVE TRAINING\nDerivative verification: {state.get('derivatives','PASSED' if v.get('successful') is not None else 'checking / preview')} | Checkpoint lookup: {'DISABLED' if self.lookup=='none' else 'EXPLICIT PATHS ONLY'}\n"
                  f"Device: {self.device_name} | Drive saving: {'ENABLED' if self._plans[0].persistence.drive_enabled else 'DISABLED in setup'} | Queries: {self.query_count:,} | Chunk: {self.query_chunk:,}\n"
                  f"Overall: {completed}/{total} models completed | {p.get('label','Preparing')}\nCurrent: {current} | Stage: {state.get('stage','STARTING')} ({time.monotonic()-since:.1f}s)\n"
                  f"Origin: {origin} | Updates: {v.get('successful',0)}/{v.get('requested',self._plans[0].settings.updates)} successful\n"
                  f"Field: {fmt(raw.get('field'))} | Auxiliary: {fmt(raw.get('auxiliary'))} | Kinematic: {fmt(raw.get('kinematic'))} | Total: {fmt(v.get('current_objective'))}\n"
                  f"Gradient: {fmt(v.get('gradient_before_clip'))} → {fmt(v.get('gradient_after_clip'))} | Clipping fraction (rolling): {fraction}\n"
                  f"Updates/s: {fmt(ups)} | Queries/s: {fmt(perf.get('queries_per_second'))} | Device allocated GB: {fmt(allocated/1e9 if allocated is not None else None)}\n"
                  f"Cell elapsed: {elapsed:.1f}s | Current-variant ETA (measured): {fmt(eta)} s | Whole-study ETA (measured projection): {fmt(whole)} s\n"
                  f"Local save: {persistence.get('local_status','not yet saved')} | Drive: {v.get('Drive_sync',persistence.get('Drive_status','not yet saved'))} | Verified: {sync.get('Drive_verified',False)}\n"
                  f"Last checkpoint: {v.get('checkpoint') or state.get('checkpoint') or 'not yet saved'}\nDrive manifest: {sync.get('Drive_manifest') or 'not yet verified'}")
            if v.get('Drive_sync')=='failed':text+='\nDRIVE SYNC FAILED — valid local checkpoint retained: '+str(sync.get('error'))
            if state.get('error'):text+=f"\nERROR during {state.get('failed_stage',self.failed_stage)} at successful update {self.last_successful}: {state['error']}"
            self.last_compact_text=text
            if self._notebook and not self._fallback:
                try:
                    from IPython.display import display,HTML
                    content=HTML('<pre style="white-space:pre-wrap;max-height:520px;overflow:auto">'+html.escape(text)+'</pre>')
                    target=previous if previous is not None else self.handle
                    if target is None:target=display(content,display_id=True)
                    else:target.update(content)
                    if target is None:raise RuntimeError('notebook display handle unavailable')
                    self.handle=target;return target
                except Exception as error:
                    self._fallback=True
                    print('Wave display refresh failed; using flushed text: '+str(error),flush=True)
            if self._fallback or self._notebook:
                print(text,flush=True)
            return previous

    def phase(self,stage,**details):
        import time
        with self._lock:
            self.phase_state={**details,'stage':stage,'since':time.monotonic()}
            if details.get('architecture'):self.architecture=details['architecture']
            if details.get('variant'):
                variant=details['variant']
                if self.architecture=='A_GEO' and variant=='FULL' and any('A_GEO/BASELINE' in r['variants'] for r in self.wave_state['preview']['experiments']):variant='BASELINE'
                label=self.architecture+'/'+variant
                if label!=getattr(self,'_preparing_label',None):
                    self._view={};self.last_successful=0;self._preparing_label=label
                self.phase_state['variant']=label
            self._emit()

    def restore_phase(self,state):
        with self._lock:self.phase_state=state or {};self._emit()

    def live(self):
        """UI-only heartbeat: never runs a model, solver, optimizer, RNG or save."""
        from contextlib import contextmanager
        from .learning_diagnostics.wave_progress import using_monitor
        import threading
        @contextmanager
        def active():
            self._stop.clear();self.failed_stage=None
            self._view={};self.last_successful=0;self._preparing_label=None;self._last_forced={}
            def tick():
                while not self._stop.wait(self._heartbeat_seconds):self._emit()
            self._heartbeat=threading.Thread(target=tick,name='wave-display-heartbeat',daemon=True)
            self._heartbeat.start()
            try:
                with using_monitor(self):yield self
            finally:
                self._stop.set();self._heartbeat.join(timeout=1.)
        return active()

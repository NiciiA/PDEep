"""Shared successful-update engine; scientific modules/objectives stay unchanged."""
from dataclasses import asdict
from pathlib import Path
import json,math,tempfile,time,traceback
import torch
from .core import require
from .train_types import TrainingState,StepReport
from .train_registry import registration,ScientificAdapter,capability_probe,tensor_summary
from .train_data import TrainingSampler,make_validation_manifest
from .train_monitor import LiveMonitor,memory_report
from .train_validation import validate
from .representation_checks import sync


def json_safe(value):
    if isinstance(value,float) and not math.isfinite(value):return str(value)
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [json_safe(v) for v in value]
    return value


class TrainingFailure(RuntimeError):
    def __init__(self,message,*,diagnostic,step):super().__init__(message);self.diagnostic=diagnostic;self.step=step


class SuccessfulScheduler:
    """LR for NEXT update; counter advances only after optimizer success.

    Warmup uses i/W for successful-update ordinal i=1..W. Then cosine reaches
    minimum at i=budget. Constant smoke schedule has no warmup. No microbatch ticks.
    """
    def __init__(self,optimizer,config):
        self.optimizer=optimizer;self.config=config;self.completed=0;self._set()

    def _set(self):
        c=self.config;i=min(self.completed+1,c.updates)
        if c.scheduler=='constant':factor=1.
        elif c.warmup_updates and i<=c.warmup_updates:factor=i/c.warmup_updates
        else:
            progress=min(1.,max(0.,(i-c.warmup_updates)/max(c.updates-c.warmup_updates,1)))
            factor=c.minimum_lr_fraction+(1-c.minimum_lr_fraction)*.5*(1+math.cos(math.pi*progress))
        for group in self.optimizer.param_groups:group['lr']=c.learning_rate*factor

    def advance(self,successful):
        require(successful==self.completed+1,'scheduler must advance exactly once per successful optimizer update')
        self.completed=successful;self._set()

    def state_dict(self):return {'completed':self.completed,'kind':self.config.scheduler,'next_lr':self.optimizer.param_groups[0]['lr']}
    def load_state_dict(self,state):
        require(state['kind']==self.config.scheduler,'scheduler/config mismatch');self.completed=state['completed'];self._set()
        require(abs(self.optimizer.param_groups[0]['lr']-state['next_lr'])<1e-15,'scheduler LR reload mismatch')


class TrainingEngine:
    def __init__(self,config,normalizers,run_root,*,monitor=None,model=None,adapter=None,probe=True,allow_existing_run=False,persistence=None):
        cap=registration(config.architecture);require(cap.trainable,'D cannot enter a neural optimizer loop')
        self.config=config;self.normalizers=normalizers;self.state=TrainingState(config.architecture,config.run_id)
        self.monitor=LiveMonitor(enabled=False) if monitor is None else monitor
        self.run_dir=Path(run_root)/config.architecture/config.run_id
        require(allow_existing_run or not self.run_dir.exists(),'run directory exists; explicitly resume or select a new run_tag')
        self.run_dir.mkdir(parents=True,exist_ok=True);self._started=time.monotonic();self._elapsed_base=0.
        self.latest_step=None;self.latest_validation=None
        from .artifact_drive import PersistencePolicy
        self.persistence_policy=PersistencePolicy() if persistence is None else persistence
        self.rng_generators={};self.parent_checkpoints=[];self.scaler=None
        self.best_validation_model_state=None
        self.state.phase='constructing';self._notify()
        if model is None:
            from .models import select_architecture
            model=select_architecture(config.architecture,normalizers=normalizers,root_seed=config.root_seed).to(config.device)
        self.model=model;self.adapter=ScientificAdapter(config.architecture) if adapter is None else adapter
        self.sampler=TrainingSampler(config,normalizers)
        self.state.phase='capability_probe';self._notify()
        try:
            self.capability=capability_probe(model,config.architecture,normalizers,config.device) if probe else {'skipped':'explicit testing injection or already verified checkpoint reload'}
        except Exception as error:
            self.state.phase='failed';self.state.last_error=f'capability probe before optimizer initialization: {error}'
            packet={'kind':'startup_capability_failure','config':asdict(config),'state':asdict(self.state),'traceback':traceback.format_exc(),'normalizer_identity':normalizers.identity}
            (self.run_dir/'startup_failure.json').write_text(json.dumps(json_safe(packet),indent=2)+'\n')
            self._notify();raise
        self.optimizer=torch.optim.AdamW(model.parameters(),lr=config.learning_rate,weight_decay=config.weight_decay)
        self.scheduler=SuccessfulScheduler(self.optimizer,config)
        self.state.phase='validation_manifest';self._notify();self.manifest=make_validation_manifest(config,normalizers)
        self.state.rng_sampler_state=self.sampler.state_dict(0);self.state.scheduler_state=self.scheduler.state_dict()
        self.state.optimizer_state={'family':'AdamW','learning_rates':[g['lr'] for g in self.optimizer.param_groups],'state_entries':0,'tensor_state':'engine.optimizer.state_dict(); persisted in every checkpoint'}
        self.state.phase='ready';self._notify()
        if not allow_existing_run:
            (self.run_dir/'config.json').write_text(json.dumps(asdict(config),indent=2)+'\n')
            (self.run_dir/'validation_manifest.json').write_text(json.dumps({'identity':self.manifest.identity,**asdict(self.manifest)},indent=2)+'\n')
        if torch.device(config.device).type=='cuda':torch.cuda.reset_peak_memory_stats(config.device)

    def _notify(self,step=None,validation=None):
        self.state.elapsed_seconds=self._elapsed_base+time.monotonic()-self._started
        if step is not None:self.latest_step=step
        if validation is not None:self.latest_validation=validation
        return self.monitor.update(self.config,self.state,self.latest_step,self.latest_validation)

    def _optimizer_step(self):
        # No GradScaler/AMP. A future scaler must return False when it skipped.
        self.optimizer.step();return True

    def _log(self,name,value):
        with (self.run_dir/name).open('a') as stream:stream.write(json.dumps(json_safe(value),allow_nan=False)+'\n')

    def update(self,batch=None):
        """One complete accumulation window; returns StepReport or raises TrainingFailure.

        An explicit batch is useful for deterministic arithmetic tests. Scientific
        runs obtain one immutable keyed batch, sliced by physical-case count.
        """
        require(self.state.phase not in {'failed','complete'},'training run stopped; explicit reload/new configuration required')
        require(self.state.successful<self.config.updates,'successful-update budget exhausted; choose an explicit new experiment rather than extending a saved schedule silently')
        require(not self.state.accumulation_active,'unfinished accumulation window')
        self.state.attempted+=1;self.state.accumulation_active=True;self.state.accumulation_microbatches=0;self.state.accumulation_cases=0
        self.state.phase='data';self._notify();timings=dict(data=0.,forward=0.,backward=0.,optimizer=0.)
        total=0.;unweighted={};weighted={};meanings={};counts={'cases':0,'valid_queries':0,'active_coefficients':0,'valid_observations':0};details={};case_ids=()
        lr=self.optimizer.param_groups[0]['lr'];before=after=None;operation='data';first_component=None;last_loss=None
        try:
            started=time.monotonic()
            if batch is None:batch,identities=self.sampler.batch(self.state.sampler_cursor);details['identities']=identities
            case_ids=batch.case_ids
            self.state.sampler_cursor+=1;self.state.rng_sampler_state=self.sampler.state_dict(self.state.sampler_cursor)
            timings['data']=time.monotonic()-started;cases=len(batch.specs);require(cases>0,'empty update window')
            self.model.train();self.optimizer.zero_grad(set_to_none=True)
            for lo in range(0,cases,self.config.microbatch):
                hi=min(cases,lo+self.config.microbatch);micro=batch.slice(lo,hi);weight=(hi-lo)/cases
                self.state.phase='forward';operation='forward';self._notify()
                sync(torch.device(self.config.device));started=time.monotonic()
                loss=self.adapter.loss(self.model,micro,detailed=self.config.verbosity=='detailed');last_loss=loss
                sync(torch.device(self.config.device));timings['forward']+=time.monotonic()-started
                self.state.cases_processed+=hi-lo;self.state.queries_processed+=int(micro.queries.valid.sum());self.state.microbatches_processed+=1
                self.state.accumulation_microbatches+=1;self.state.accumulation_cases+=hi-lo
                for spec in micro.specs:self.state.pde_counts[spec.name]=self.state.pde_counts.get(spec.name,0)+1
                for group in (loss.unweighted,loss.weighted,{'total':loss.total}):
                    for component,value in group.items():
                        if not bool(torch.isfinite(value).all()):first_component=component;raise FloatingPointError(f'nonfinite objective component {component}')
                operation='backward';self.state.phase='backward';self._notify();started=time.monotonic()
                (weight*loss.total).backward();sync(torch.device(self.config.device));timings['backward']+=time.monotonic()-started
                total+=weight*float(loss.total.detach())
                for key,value in loss.unweighted.items():unweighted[key]=unweighted.get(key,0.)+weight*float(value.detach())
                for key,value in loss.weighted.items():weighted[key]=weighted.get(key,0.)+weight*float(value.detach())
                meanings.update(loss.meanings)
                for key,value in loss.counts.items():counts[key]=counts.get(key,0)+value
                if self.config.verbosity=='detailed':details.setdefault('microbatches',[]).append(loss.diagnostics)
            operation='gradient_inspection'
            for name,p in self.model.named_parameters():
                if p.grad is not None and not bool(torch.isfinite(p.grad).all()):first_component=name;raise FloatingPointError(f'nonfinite parameter gradient {name}')
            if self.config.verbosity=='detailed':
                from .a_training import gradient_report
                details['parameter_gradients']=gradient_report(self.model)
            # Clip once, after the complete case-weighted window.
            operation='gradient_clipping'
            before=float(torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.config.grad_clip_norm,error_if_nonfinite=True))
            after=sum(float(p.grad.detach().square().sum()) for p in self.model.parameters() if p.grad is not None)**.5
            operation='optimizer';self.state.phase='optimizer';self._notify();started=time.monotonic()
            performed=self._optimizer_step();sync(torch.device(self.config.device));timings['optimizer']=time.monotonic()-started
            if not performed:
                self.state.skipped+=1;status='skipped'
            else:
                require(all(bool(torch.isfinite(p).all()) for p in self.model.parameters()),'nonfinite parameter after optimizer')
                self.state.successful+=1;operation='scheduler';self.scheduler.advance(self.state.successful);status='successful'
            self.state.accumulation_active=False;self.state.accumulation_microbatches=0;self.state.accumulation_cases=0
            self.state.scheduler_state=self.scheduler.state_dict();self.state.phase='ready'
            self.state.optimizer_state.update(learning_rates=[g['lr'] for g in self.optimizer.param_groups],state_entries=len(self.optimizer.state),successful_updates=self.state.successful)
            report=StepReport(self.config.architecture,self.state.run_id,self.state.attempted,self.state.successful,self.state.skipped,self.state.failed,status,total,unweighted,weighted,meanings,counts,lr,self.optimizer.param_groups[0]['lr'],before,after,timings,True,memory_report(self.config.device),tuple(case_ids),details)
            if performed:self.state.recent_steps=(self.state.recent_steps+[asdict(report)])[-self.config.rolling_window:]
            self._log('steps.jsonl',asdict(report));self._notify(report);return report
        except Exception as error:
            self.state.failed+=1;self.state.phase='failed';self.state.last_error=f'{operation}: {type(error).__name__}: {error}'
            kind='out_of_memory' if isinstance(error,torch.OutOfMemoryError) or 'out of memory' in str(error).lower() else 'unsupported_operation' if isinstance(error,NotImplementedError) else 'nonfinite' if isinstance(error,FloatingPointError) or 'nonfinite' in str(error).lower().replace('-','') else 'implementation_or_data_error'
            if kind=='nonfinite' and first_component is None and operation=='gradient_clipping':first_component='global_gradient_norm'
            packet={'kind':kind,'operation':operation,'first_component':first_component,'error':str(error),'traceback':traceback.format_exc(),
                    'config':asdict(self.config),'state':asdict(self.state),'case_ids':case_ids,'recent_history':self.state.recent_steps,
                    'microbatch':self.config.microbatch,'C_quadrature':{'spatial':16,'time':8},'memory':memory_report(self.config.device),'objective_values':unweighted}
            if last_loss is not None:
                packet['current_loss_components']={k:float(v.detach()) for k,v in last_loss.unweighted.items()}
            if batch is not None:
                packet['pdes']=[s.name for s in batch.specs]
                packet['inputs']={name:tensor_summary(getattr(batch.observations,name),batch.observations.valid if name!='physical_time' else batch.observations.valid&batch.observations.time_present) for name in ('position','physical_time','value','sigma')}
            packet['invalid_gradients']=[name for name,p in self.model.named_parameters() if p.grad is not None and not bool(torch.isfinite(p.grad).all())]
            directory=self.run_dir/'failures';directory.mkdir(exist_ok=True)
            with tempfile.NamedTemporaryFile(mode='w',dir=directory,prefix=f'attempt-{self.state.attempted}-',suffix='.json',delete=False) as stream:
                json.dump(json_safe(packet),stream,indent=2,allow_nan=False);diagnostic=stream.name
            try:
                from .diagnostic_artifacts import save_failure_bundle
                packet['complete_diagnostic_bundle']=str(save_failure_bundle(self,packet,batch))
            except Exception as diagnostic_error:
                packet['diagnostic_bundle_error']=str(diagnostic_error)
            from .artifact_io import atomic_json
            atomic_json(diagnostic,json_safe(packet))
            self._log('failures.jsonl',json_safe(packet))
            report=StepReport(self.config.architecture,self.state.run_id,self.state.attempted,self.state.successful,self.state.skipped,self.state.failed,'failed',None,unweighted,weighted,meanings,counts,lr,self.optimizer.param_groups[0]['lr'],before,after,timings,False,memory_report(self.config.device),tuple(case_ids),{'failure_diagnostic':diagnostic,'complete_diagnostic_bundle':packet.get('complete_diagnostic_bundle')},self.state.last_error)
            self._log('steps.jsonl',asdict(report));self._notify(report)
            raise TrainingFailure(self.state.last_error,diagnostic=diagnostic,step=report) from error

    def validate(self):
        require(not self.state.accumulation_active,'validation only at a completed window boundary')
        self.state.phase='validation';self._notify()
        try:
            report=validate(self.model,self.config.architecture,self.manifest,self.normalizers,device=self.config.device,successful_update=self.state.successful)
            metric=report.metrics['common']['field_normalized_MSE']
            require(math.isfinite(metric),'best-validation metric must be finite')
            self.state.validation_seconds+=report.elapsed_seconds;self.state.validation_history.append({'step':self.state.successful,'manifest':self.manifest.identity,'checkpoint':report.checkpoint_identity,'metric':self.state.validation_metric_name,'value':metric})
            if self.state.successful>0 and (self.state.best_validation_metric is None or metric<self.state.best_validation_metric):
                self.state.best_validation_metric=metric;self.state.best_validation_step=self.state.successful;self.state.best_validation_weights=report.checkpoint_identity
                from .artifact_tensors import tree_cpu
                self.best_validation_model_state=tree_cpu(self.model.state_dict())
            self._log('validation.jsonl',asdict(report));return report
        except Exception as error:
            self.state.last_error='validation diagnostic: '+str(error);raise
        finally:
            self.state.phase='ready';self._notify(validation=locals().get('report'))

    def save_checkpoint(self):
        from .train_checkpoint import save_checkpoint
        require(not self.state.accumulation_active and self.state.phase!='failed','checkpoint requires a successful optimizer boundary')
        self.state.phase='saving';self._notify();start=time.monotonic()
        try:
            path=save_checkpoint(self)
            self.state.last_checkpoint=str(path);self.state.checkpoint_history.append({'step':self.state.successful,'path':str(path)})
            return path
        except Exception as error:
            self.state.last_error='checkpoint diagnostic: '+str(error);raise
        finally:self.state.saving_seconds+=time.monotonic()-start;self.state.phase='ready';self._notify()

    def run(self,*,launch_full=False):
        c=self.config
        require(c.preset not in {'research','large'} or c.full_training_enabled and launch_full,'full training disabled; requires config.full_training_enabled AND deliberate run(launch_full=True)')
        require(self.state.phase!='failed','failed run cannot silently resume')
        reports=[]
        while self.state.successful<c.updates:
            step=self.update();reports.append(step)
            require(self.state.attempted<=c.updates+100,'too many skipped attempts; stopping safely')
            if step.status=='successful':
                if self.state.successful%c.validation_every==0:
                    self.validate()
                    if self.state.best_validation_step==self.state.successful:self.save_checkpoint()
                if self.state.successful%c.checkpoint_every==0 and not (self.state.checkpoint_history and self.state.checkpoint_history[-1]['step']==self.state.successful):self.save_checkpoint()
        if not self.state.validation_history or self.state.validation_history[-1]['step']!=self.state.successful:self.validate()
        if not self.state.checkpoint_history or self.state.checkpoint_history[-1]['step']!=self.state.successful:self.save_checkpoint()
        self.state.phase='complete';self._notify(reports[-1] if reports else None)
        return reports

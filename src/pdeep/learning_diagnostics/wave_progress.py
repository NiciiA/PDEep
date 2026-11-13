"""Optional observation hooks; no numerical work, RNG or optimizer changes."""
from contextlib import contextmanager
from contextvars import ContextVar

_MONITOR=ContextVar('wave_training_monitor',default=None)

@contextmanager
def using_monitor(monitor):
    token=_MONITOR.set(monitor)
    try:yield
    finally:_MONITOR.reset(token)

@contextmanager
def phase(stage,**details):
    monitor=_MONITOR.get()
    previous=None if monitor is None else monitor.phase_state.copy()
    if monitor is not None:monitor.phase(stage,**details)
    try:yield
    except BaseException:
        if monitor is not None:monitor.failed_stage=stage
        raise
    finally:
        if monitor is not None:
            monitor.restore_phase(previous)

def observe(stage,fn,**details):
    with phase(stage,**details):return fn()

"""Visible linear notebook state/provenance guards for safe out-of-order reruns."""
from contextlib import contextmanager
from dataclasses import is_dataclass
from pathlib import Path
from .core import digest,require
from .diagnostics import source_identity

TRACKED=('CONFIG','NORMALIZERS','TRAIN_CONFIGS','EVAL_CONFIG','EVAL_PLAN')


def notebook_bindings(namespace):
    result={}
    for name in TRACKED:
        if name not in namespace:continue
        obj=namespace[name]
        if isinstance(obj,dict):result[name]=digest({k:v.identity for k,v in obj.items()})
        else:result[name]=obj.identity
    return result


class NotebookLedger:
    """Rerunning a cell invalidates later cells; changed configs invalidate owner.

    Completed-cell stamps and binding ownership are inspectable via snapshot().
    Source checking excludes notebook outputs. No training or artifact mutation.
    """
    def __init__(self,root,source_id,order):
        self.root=Path(root);self.source_id=source_id;self.order=tuple(order);self.completed={};self.bindings={};self.generation=0

    def _invalidate(self,cell):
        start=self.order.index(cell)
        for later in self.order[start:]:self.completed.pop(later,None)

    @contextmanager
    def cell(self,cell,namespace):
        require(cell in self.order,'unknown notebook cell identity')
        require(source_identity(self.root)['sha256']==self.source_id,'notebook source changed; rerun section01 and dependent cells')
        index=self.order.index(cell)
        if index:require(self.order[index-1] in self.completed,'required previous notebook state missing/invalidated; execute cells in dependency order')
        current=notebook_bindings(namespace)
        for name,(owner,identity) in list(self.bindings.items()):
            if current.get(name)!=identity and owner!=cell:
                self._invalidate(owner)
                raise ValueError(f'{name} configuration/normalization changed; rerun its owning cell {owner} and downstream cells')
        self._invalidate(cell)
        try:yield
        except Exception:
            self._invalidate(cell);raise
        else:
            after=notebook_bindings(namespace)
            for name,identity in after.items():
                if name not in self.bindings or self.bindings[name][1]!=identity:self.bindings[name]=(cell,identity)
            self.generation+=1
            self.completed[cell]={'source_identity':self.source_id,'configuration_identities':after,'generation':self.generation}

    def snapshot(self):return {'source_identity':self.source_id,'completed_cells':len(self.completed),'generation':self.generation,'bindings':{k:{'owner':v[0],'identity':v[1]} for k,v in self.bindings.items()}}

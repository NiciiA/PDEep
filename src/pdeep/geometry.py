"""Geometry-owned anchor placement and boundary features, independent of neural layers.

The canonical physics remains the unit square. Other providers can supply
representation geometry without claiming support for another physical problem.
"""
from dataclasses import dataclass
from typing import Protocol
import math
import torch

from .core import require


@dataclass(frozen=True)
class AnchorLayout:
    positions: torch.Tensor
    levels: torch.Tensor  # 0 coarse, 1 fine; storage need not be contiguous

    def validate(self):
        require(self.positions.ndim==2 and self.positions.shape[1]==2,"anchor positions must be [M,2]")
        require(self.positions.dtype==torch.float32 and bool(torch.isfinite(self.positions).all()),"anchor layout must be finite float32")
        require(self.levels.shape==self.positions.shape[:1] and self.levels.dtype==torch.int64,"anchor levels must be int64 [M]")
        require(bool(((self.levels==0)|(self.levels==1)).all()) and bool((self.levels==0).any())
                and bool((self.levels==1).any()),"geometry must declare nonempty coarse/fine sets")
        return self


class GeometryProvider(Protocol):
    boundary_feature_size: int

    def anchor_layout(self) -> AnchorLayout: ...
    def boundary_features(self,position: torch.Tensor) -> torch.Tensor: ...
    def boundary_distance(self,position: torch.Tensor) -> torch.Tensor: ...
    def neighbor_weights(self) -> torch.Tensor: ...


@dataclass(frozen=True)
class BoxGeometry:
    bounds: tuple[tuple[float,float],tuple[float,float]] = ((0.,1.),(0.,1.))
    coarse_side: int = 4
    fine_side: int = 8
    boundary_feature_size: int = 4

    def __post_init__(self):
        require(len(self.bounds)==2 and all(len(b)==2 and all(math.isfinite(x) for x in b) and b[0]<b[1] for b in self.bounds),"invalid box bounds")
        require(type(self.coarse_side) is int and type(self.fine_side) is int and min(self.coarse_side,self.fine_side)>0,"invalid anchor grid sizes")
        require(self.boundary_feature_size==4,"box geometry has four boundary features")

    def boundary_features(self,position):
        lower=position.new_tensor([b[0] for b in self.bounds])
        upper=position.new_tensor([b[1] for b in self.bounds])
        return torch.cat((position-lower,upper-position),dim=-1)

    def anchor_layout(self):
        positions,levels=[],[]
        for level,side in enumerate((self.coarse_side,self.fine_side)):
            axes=[lo+(hi-lo)*(torch.arange(side,dtype=torch.float32)+.5)/side for lo,hi in self.bounds]
            yy,xx=torch.meshgrid(axes[1],axes[0],indexing="ij")
            positions.append(torch.stack((xx,yy),dim=-1).reshape(-1,2))
            levels.append(torch.full((side*side,),level,dtype=torch.int64))
        return AnchorLayout(torch.cat(positions),torch.cat(levels)).validate()

    def boundary_distance(self,position):
        return self.boundary_features(position).min(-1,keepdim=True).values

    def neighbor_weights(self):
        """Nonperiodic fine axial neighbors, all other coarse, nearest coarse reads.

        Row=receiver, column=sender. No self edges. Fine reads four nearest
        coarse anchors (ties deterministic by storage index). Rows sum to one.
        This discrete geometry is fixed independently of future queries.
        """
        layout=self.anchor_layout();p,level=layout.positions,layout.levels
        delta=(p[:,None]-p[None]).abs()
        spacing=p.new_tensor([(hi-lo)/self.fine_side for lo,hi in self.bounds])
        fine=(level==1);coarse=~fine
        axial=((delta[...,0]<1e-6)&((delta[...,1]-spacing[1]).abs()<1e-6)) | ((delta[...,1]<1e-6)&((delta[...,0]-spacing[0]).abs()<1e-6))
        edges=axial & fine[:,None] & fine[None,:]
        edges |= coarse[:,None]&coarse[None,:]&~torch.eye(len(p),dtype=torch.bool)
        ids=torch.where(coarse)[0]
        nearest=torch.argsort((p[:,None]-p[ids][None]).square().sum(-1),dim=-1,stable=True)[:,:min(4,len(ids))]
        for row in torch.where(fine)[0]:edges[row,ids[nearest[row]]]=True
        weights=edges.to(torch.float32)
        return weights/weights.sum(-1,keepdim=True).clamp_min(1)


class SquareDirichletTreatment:
    """Explicit V1 boundary treatment; no claim of arbitrary-geometry support."""
    def __init__(self,geometry):
        require(isinstance(geometry,BoxGeometry) and geometry.bounds==((0.,1.),(0.,1.)),
                "A boundary treatment supports the unit square only")

    def factor(self,position):
        return (position*(1-position)).prod(-1,keepdim=True)


def geometry_for(spec_geometry=None):
    if spec_geometry is not None:
        require(spec_geometry.kind=="open_unit_square" and spec_geometry.bounds==((0.,1.),(0.,1.)),
                "physical geometry adapter currently supports the canonical unit square only")
    return BoxGeometry()


def boundary_features(position,provider=None):
    return (geometry_for() if provider is None else provider).boundary_features(position)

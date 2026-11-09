"""Review figures for continuity diagnostics; no learned prediction results."""
import numpy as np
import torch
import matplotlib.pyplot as plt

from .geometry import geometry_for
from .memory import gaussian_readout


def retrieval_review_figure(kernels,neural,legacy):
    fig,axes=plt.subplots(2,2,figsize=(13,9),layout="constrained")
    old=legacy["samples"]
    axes[0,0].loglog([s["epsilon"] for s in old],[max(s["feature_jump_per_pde"]) for s in old],"o-",label="old hard top-k (measured before change)")
    axes[0,0].loglog([s["epsilon"] for s in neural["transition_continuity"]],
                     [max(s["feature_difference_per_pde"]) for s in neural["transition_continuity"]],"o-",label="new all-anchor Gaussian")
    axes[0,0].invert_xaxis()
    axes[0,0].set(xlabel="epsilon: x = 0.5 ± epsilon",ylabel="max fused-feature difference",title="Fixed-memory neural feature continuity (float32)")
    axes[0,0].legend(fontsize=8)
    layout=geometry_for().anchor_layout()
    position=torch.tensor([[[.5,.5]]],dtype=torch.float64)
    for level,ell,color in ((0,.30,"#237da0"),(1,.15,"#c76527")):
        anchors=layout.positions[layout.levels==level].double()
        token=(13*anchors[:,0]).sin()+(7*anchors[:,1]).cos()
        x=torch.linspace(.4,.6,201,dtype=torch.float64)
        p=torch.stack((x,torch.full_like(x,.5)),dim=-1)[None].requires_grad_()
        value,weights=gaussian_readout(p,anchors,token[None,:,None],ell)
        derivative=torch.autograd.grad(value.sum(),p)[0][0,:,0]
        axes[0,1].plot(x.numpy(),derivative.detach().numpy(),label=f"{'coarse' if level==0 else 'fine'}, ell={ell}",color=color)
        series=kernels["scales"][level]["continuity"]
        axes[1,0].loglog([s["epsilon"] for s in series],[s["derivative_difference"] for s in series],"o-",label="coarse" if level==0 else "fine",color=color)
        _,center_weights=gaussian_readout(position,anchors,token[None,:,None],ell)
        ids=np.arange(len(anchors))+(16 if level else 0)
        axes[1,1].bar(ids,center_weights[0,0].numpy(),color=color,label=f"{'coarse' if level==0 else 'fine'} sum = 1")
    axes[0,1].axvline(.5,color="gray",ls="--")
    axes[0,1].set(xlabel="query x at y=0.5",ylabel="d(readout)/dx",title="Continuous kernel derivative across old switch")
    axes[0,1].legend()
    axes[1,0].invert_xaxis()
    axes[1,0].set(xlabel="epsilon",ylabel="max derivative difference, ±epsilon",title="Float64 first-derivative continuity at 15 probes")
    axes[1,0].legend()
    axes[1,1].axvline(15.5,color="gray",ls="--")
    axes[1,1].set(xlabel="anchor index, coarse then fine",ylabel="Gaussian weight",title="All 80 weights; independent scale normalization")
    axes[1,1].legend()
    fig.suptitle("Stage 3 retrieval review — feature diagnostics only; Stage 4 not started")
    return fig

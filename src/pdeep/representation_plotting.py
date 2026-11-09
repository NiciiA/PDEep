"""Compact untrained-representation figures for the research notebook."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import torch


def representation_diagram():
    fig,ax=plt.subplots(figsize=(15,7),layout="constrained")
    ax.set(xlim=(0,15),ylim=(0,7));ax.axis("off")
    def box(x,y,w,h,text,color):
        patch=FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.12",facecolor=color,edgecolor="#334957",linewidth=1.2)
        ax.add_patch(patch);ax.text(x+w/2,y+h/2,text,ha="center",va="center",fontsize=10)
    def arrow(a,b):
        ax.annotate("",xy=b,xytext=a,arrowprops={"arrowstyle":"->","color":"#334957","lw":1.5})
    box(.3,5.1,3.,1.1,"Validated symbolic DAG\ntypes • identities • numeric values\nordered operands + masks","#e6eef7")
    box(4.,5.1,3.3,1.1,"Equation encoder\n2 typed bidirectional layers\nwidth 96","#d9e9f5")
    box(8.1,5.1,3.5,1.1,"Nodes [B,V,96]\nEquation context [B,96]","#e8f3ee")
    arrow((3.45,5.65),(3.85,5.65));arrow((7.45,5.65),(7.95,5.65))
    box(.3,3.2,3.,1.1,"Scalar observations\ncoordinates + optional time\nquantity • value • sigma • mask","#e6eef7")
    box(4.,3.2,3.3,1.1,"Observation encoder\nFourier + training RMS + log noise\npointwise MLP + masked pooling","#d9e9f5")
    box(8.1,3.2,3.5,1.1,"Tokens [B,N,96]\nObservation context [B,96]","#e8f3ee")
    arrow((3.45,3.75),(3.85,3.75));arrow((7.45,3.75),(7.95,3.75))
    box(.3,.6,4.1,1.3,"B-compatible coefficient tokens\nmode • block • xi • prior scale\ngradient • previous update • active\n[B,18,96]; no field memory required","#fff0d9")
    box(5.2,.6,4.,1.3,"Optional spatial memory\n16 coarse + 64 fine interior anchors\n2 cross-attention layers, 4 heads\nknown context + coarse/fine mixing","#d9e9f5")
    box(10.1,.6,4.2,1.3,"Conditional coordinate/query features\nGaussian: all 16 coarse + 64 fine\nseparate scale averages + global\n[B,Q,96]; no prediction head","#e8f3ee")
    arrow((9.8,3.05),(7.2,2.05));arrow((10.,4.95),(13.,2.05));arrow((9.35,1.25),(9.95,1.25))
    ax.text(7.5,6.7,"Independent parameter instances for A, B and C",ha="center",fontsize=15,weight="bold")
    ax.text(7.5,.05,"Representation modules only — untrained diagnostics, no A/B/C prediction architecture",ha="center",fontsize=11)
    return fig


def embedding_examples(equation,observation,specs):
    fig,axes=plt.subplots(2,1,figsize=(14,6),layout="constrained")
    array=equation.context.detach().cpu().numpy()
    image=axes[0].imshow(array,aspect="auto",cmap="RdBu_r",vmin=-float(np.abs(array).max()),vmax=float(np.abs(array).max()))
    axes[0].set(yticks=np.arange(len(specs)),yticklabels=[s.name for s in specs],xlabel="latent channel (arbitrary learned basis)",title="Example equation-context embeddings [B,96]")
    fig.colorbar(image,ax=axes[0],shrink=.8)
    for i,spec in enumerate(specs):
        mask=observation.valid[i].detach().cpu().numpy()
        token=observation.tokens[i].detach().cpu().numpy()
        # One feature channel shows token variation without printing a tensor.
        axes[1].plot(np.flatnonzero(mask),token[mask,0],".-",label=spec.name,alpha=.8)
    axes[1].set(xlabel="observation storage index (not a positional feature)",ylabel="token channel 0",title="Observation token examples; padding omitted")
    axes[1].legend()
    fig.suptitle("Untrained feature examples — no scientific predictions or inferred physical importance")
    return fig


def memory_attention(state,observation,*,row=2):
    fig,axes=plt.subplots(2,2,figsize=(13,10),layout="constrained")
    positions=state.positions.detach().cpu().numpy()
    axes[0,0].scatter(positions[:16,0],positions[:16,1],marker="s",s=100,facecolors="none",edgecolors="#b74636",label="coarse 4×4")
    axes[0,0].scatter(positions[16:,0],positions[16:,1],marker="o",s=25,color="#287da1",label="fine 8×8")
    axes[0,0].set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",title="80 interior memory anchors")
    axes[0,0].set_aspect("equal");axes[0,0].legend()
    attention=state.attention[row].detach().cpu().numpy().mean((0,1))
    xy=observation.position[row].detach().cpu().numpy()
    valid=observation.valid[row].detach().cpu().numpy()
    mean=attention.mean(0)
    image=axes[0,1].scatter(xy[valid,0],xy[valid,1],c=mean[valid],cmap="viridis",s=40)
    axes[0,1].set(xlim=(0,1),ylim=(0,1),xlabel="x",ylabel="y",title="Mean untrained sensor-attention weight")
    fig.colorbar(image,ax=axes[0,1],shrink=.8)
    image=axes[1,0].imshow(attention,aspect="auto",origin="lower",cmap="viridis")
    axes[1,0].axhline(15.5,color="white",ls="--")
    axes[1,0].set(xlabel="observation index",ylabel="anchor index (16 coarse, 64 fine)",title="Attention averaged across two layers / four heads")
    fig.colorbar(image,ax=axes[1,0],shrink=.8)
    entropy=-(attention*np.log(np.maximum(attention,np.finfo(np.float32).tiny))).sum(-1)
    axes[1,1].plot(np.arange(16),entropy[:16],"s",label="coarse")
    axes[1,1].plot(np.arange(16,80),entropy[16:],".",label="fine")
    axes[1,1].set(xlabel="anchor index",ylabel="entropy (nats)",title="Attention spread over valid observations")
    axes[1,1].legend()
    fig.suptitle("Spatial memory diagnostics — random initialization, no physical solver or learned result")
    return fig

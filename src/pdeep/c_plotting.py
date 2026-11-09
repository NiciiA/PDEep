"""C-specific neural/weak/strong diagnostics; clearly labeled by training state."""
from dataclasses import replace
import torch
import matplotlib.pyplot as plt
from .a_checks import mixed_fixture,slice_tokens
from .constraints import physical_losses,strong_diagnostics
from .c_checks import independent_diagnostic_queries
from .data import CaseDataset,observations,case_coefficients,evaluation_grid
from .reference import canonical_spec,evaluate_batch
from .representations import float32_batch


def architecture_diagram():
    fig,ax=plt.subplots(figsize=(12,4));ax.axis('off')
    entries=[(.13,.76,'Typed DAG + known numbers\n2 layers × 96'),(.13,.23,'Noisy observation tokens\n2 residual self-attention blocks'),
             (.45,.5,'Masked set pooling + graph\nknown physics + geometry\nGlobal context [B,96]'),
             (.79,.75,'Pointwise FiLM field\n4 × 128 SiLU + u/v heads\ntraining RMS × boundary'),
             (.79,.23,'xi head → source / initial fields\n9 weak tests; 16² space × 8 times\nSeparate initial t=0 consistency')]
    for x,y,s in entries:ax.text(x,y,s,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.6',fc='#edf9f1',ec='#417c55'))
    for a,b in [((.26,.75),(.33,.57)),((.26,.23),(.33,.43)),((.58,.55),(.64,.72)),((.58,.45),(.64,.25)),((.79,.62),(.79,.36))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->'))
    ax.text(.43,.04,'No anchors or solver in C. Query points do not interact.\nBoundary is hard; PDE and initial constraints are soft.',ha='center',fontsize=9)
    fig.tight_layout();return fig


def constraint_summary(model,normalizers,*,status='untrained'):
    device=next(model.parameters()).device;dev=device.type if device.type in ('cpu','mps') else str(device)
    b=mixed_fixture(normalizers,dev,queries=16);context=model.prepare_context(b.specs,b.observations)
    p=physical_losses(model,context,create_graph=False)
    fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    residuals=[(0,0),(1,0),(2,0),(2,1)]
    for col,(i,k) in enumerate(residuals):
        r=p.cases[i]['residuals'][k];scale=p.cases[i]['scales'][k]
        axes[0,col].plot(range(1,10),(r/scale).detach().square().mean(0).sqrt().cpu(),'o-')
        axes[0,col].set(title=f'{b.specs[i].name}: R{k+1}',xlabel='Lexicographic test mode',ylabel='Normalized RMS (time nodes)')
    axes[1,0].text(.5,.5,'Poisson: no physical time\nNo initial loss',ha='center',va='center');axes[1,0].axis('off')
    for col,(i,k) in enumerate(((1,0),(2,0),(2,1)),1):
        difference=p.cases[i]['initial_difference'][k]
        im=axes[1,col].imshow(difference.detach().cpu().reshape(16,16),origin='lower',cmap='RdBu_r');fig.colorbar(im,ax=axes[1,col],shrink=.7)
        axes[1,col].set(title=f"{b.specs[i].name}: t=0 {('u','v')[k]} discrepancy",xlabel='Gauss node index',ylabel='Gauss node index')
    fig.suptitle(f'C — {status}; finite test space does not certify the PDE')
    return fig


def strong_maps(model,normalizers,*,status='untrained'):
    device=next(model.parameters()).device;dev=device.type if device.type in ('cpu','mps') else str(device)
    b=mixed_fixture(normalizers,dev,queries=16)
    # Shifted 13² interior diagnostic map, independently of Gauss nodes.
    axis=torch.linspace(.037,.963,13,device=device);yy,xx=torch.meshgrid(axis,axis,indexing='ij')
    p=torch.stack((xx,yy),-1).reshape(1,-1,2).expand(3,-1,-1).clone()
    valid=torch.ones(3,len(axis)**2,dtype=torch.bool,device=device);time=torch.full((3,len(axis)**2,1),.237,device=device)
    present=valid.clone();present[0]=False;time[0]=0
    from .batches import QueryBatch
    q=QueryBatch(p,time,present,torch.zeros_like(valid,dtype=torch.int64),valid)
    context=model.prepare_context(b.specs,b.observations);r=strong_diagnostics(model,context,q)
    fig,axes=plt.subplots(1,4,figsize=(13,3.5),constrained_layout=True)
    items=[('Poisson −Δu−f',r[0]['residual_values'][0]),('Heat u_t−αΔu',r[1]['residual_values'][0]),('Wave u_t−v',r[2]['residual_values'][0]),('Wave v_t−c²Δu',r[2]['residual_values'][1])]
    for ax,(label,value) in zip(axes,items):
        im=ax.imshow(value.cpu().reshape(13,13),origin='lower',extent=(.037,.963,.037,.963),cmap='RdBu_r');fig.colorbar(im,ax=ax,shrink=.7);ax.set_title(label,fontsize=10)
    fig.suptitle(f'C — {status}: independent strong residuals in physical units, t=.237')
    return fig


def training_curve(report):
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),constrained_layout=True);logs=report['updates'];x=[l['update'] for l in logs]
    for k in ('field','xi','weak','initial','total'):axes[0].plot(x,[l[k] for l in logs],'o-',label=k)
    axes[0].legend();axes[0].set(xlabel='Optimizer update s',ylabel='Unweighted normalized components / total',title=report['status'])
    for k in ('quadrature_forward_seconds','coordinate_derivative_seconds','backward_seconds'):axes[1].plot(x,[l[k] for l in logs],'o-',label=k)
    axes[1].legend(fontsize=8);axes[1].set(xlabel='Optimizer update s',ylabel='Seconds')
    return fig


def field_examples(model,normalizers,*,status='untrained',resolution=64):
    dataset=CaseDataset(normalizers.root_seed);device=next(model.parameters()).device
    dev=device.type if device.type in ('cpu','mps') else str(device)
    fig,axes=plt.subplots(4,3,figsize=(11,12),constrained_layout=True)
    for row,(name,quantity) in enumerate((('poisson',0),('heat',0),('wave',0),('wave',1))):
        case=dataset.case('train',name,7);spec=canonical_spec(name)
        obs=observations(dataset,case,normalizers,set_id='C-plot').inputs
        query=evaluation_grid(name,quantity_id=quantity,resolution=resolution,times=None if name=='poisson' else (.2,))
        truth=evaluate_batch((spec,),case_coefficients((case,)),query)[0,:,0].reshape(resolution,resolution).numpy()
        with torch.no_grad():
            ctx=model.prepare_context((spec,),float32_batch(obs,device=dev));q=float32_batch(query,device=dev)
            pred=torch.cat([model.predict_queries(ctx,slice_tokens(q,j,min(j+512,resolution**2))).predicted_value for j in range(0,resolution**2,512)],1)
        pred=pred[0,:,0].cpu().numpy().reshape(resolution,resolution);limit=max(abs(truth).max(),abs(pred).max(),1e-12)
        for col,(value,label) in enumerate(((truth,'Ground truth'),(pred,'Neural field'),(pred-truth,'Prediction − truth'))):
            im=axes[row,col].imshow(value,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-limit,vmax=limit)
            axes[row,col].set_title(f"{name} {('u','v')[quantity]} — {label}"+('' if name=='poisson' else ', t=.2'),fontsize=10)
            fig.colorbar(im,ax=axes[row,col],shrink=.7)
    fig.suptitle(f'Architecture C — {status}. Direct neural field; accuracy not established.')
    return fig


def observation_summary(model,normalizers,*,status='untrained'):
    device=next(model.parameters()).device;dev=device.type if device.type in ('cpu','mps') else str(device)
    b=mixed_fixture(normalizers,dev,queries=8)
    fig,axes=plt.subplots(2,3,figsize=(12,6),constrained_layout=True)
    for i,s in enumerate(b.specs):
        mask=b.observations.valid[i];obs=b.observations
        p=obs.position[i,mask].cpu();v=obs.value[i,mask,0].cpu();qty=obs.quantity_id[i,mask].cpu()
        im=axes[0,i].scatter(p[:,0],p[:,1],c=v,cmap='RdBu_r');fig.colorbar(im,ax=axes[0,i],shrink=.7)
        axes[0,i].set(title=f'{s.name}: supplied noisy sensor values',xlim=(0,1),ylim=(0,1),aspect='equal')
        if s.name=='poisson':axes[1,i].text(.5,.5,'Physical time absent',ha='center',va='center');axes[1,i].axis('off')
        else:
            t=obs.physical_time[i,mask,0].cpu();axes[1,i].scatter(t,qty,c=v,cmap='RdBu_r');axes[1,i].set(xlabel='Physical time t',ylabel='Quantity ID (0=u, 1=v)',xlim=(0,1),yticks=[0,1])
    fig.suptitle(f'C — {status}; context contains only supplied observations and known physics')
    return fig


def fixed_query_comparison(model,batch,*,status):
    with torch.no_grad():prediction=model(batch.specs,batch.observations,batch.queries).predicted_value
    fig,axes=plt.subplots(1,2,figsize=(9,4),constrained_layout=True)
    for q in (0,1):
        mask=batch.queries.valid&(batch.queries.quantity_id==q)
        truth=batch.target[mask].cpu().numpy().ravel();pred=prediction[mask].cpu().numpy().ravel()
        axes[q].scatter(truth,pred,s=6,alpha=.5)
        if len(truth):
            lo,hi=min(truth.min(),pred.min()),max(truth.max(),pred.max());axes[q].plot([lo,hi],[lo,hi],'k--')
        axes[q].set(title=f"Fixed training queries: {('u','v')[q]}",xlabel='Ground truth',ylabel='C physical prediction')
    fig.suptitle(status);return fig

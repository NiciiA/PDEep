"""B diagnostics: coefficients, exact physical fields and spatial-memory features."""
import torch
import matplotlib.pyplot as plt
from .b_physics import cpu64
from .data import CaseDataset,observations,evaluation_grid,case_coefficients
from .reference import canonical_spec,evaluate_batch
from .representations import float32_batch


def architecture_diagram(approach):
    fig,ax=plt.subplots(figsize=(12,3.5));ax.axis('off')
    boxes=[(.13,.7,'Typed DAG + noisy sensors\nIndependent96-wide encoders'),(.43,.7,'Amortized xi0 initializer\n18 masked physical slots'),(.79,.7,'3 coefficient-memory refinement steps\nphysical Hxi, residual and ∇Phi\npositive bounded diagonal/step')]
    if approach=='B1':boxes.append((.47,.23,'B1:16 coarse +64 fine geometry anchors\nprojected candidate fields and gradients\n2 sensor-attention/local-global blocks\nrecurrent within case → coefficient exchange'))
    else:boxes.append((.47,.23,'B0:9/18 active coefficient tokens\nmode/block, xi, prior, g, previous update\n2 masked token-attention blocks\nNo spatial anchor memory'))
    for x,y,s in boxes:ax.text(x,y,s,ha='center',va='center',fontsize=9,bbox=dict(boxstyle='round,pad=.6',fc='#fff4e5',ec='#967044'))
    for a,b in [((.26,.7),(.32,.7)),((.56,.7),(.64,.7)),((.49,.42),(.69,.57))]:ax.annotate('',xy=b,xytext=a,arrowprops=dict(arrowstyle='->'))
    ax.text(.83,.12,'Final xi → validated CPU64\nphysical solution S_P(xi)\nNo learned field decoder',ha='center',fontsize=9)
    fig.suptitle(approach+' — learned inverse refinement, physical forward solution');fig.tight_layout();return fig


def coefficient_trajectory(results,batch,*,status):
    fig,axes=plt.subplots(len(results),3,figsize=(12,3*len(results)),squeeze=False,constrained_layout=True)
    for row,(name,r) in enumerate(results.items()):
        for i,s in enumerate(batch.specs):
            d=int(r.xi_active[i].sum());z=((r.trajectory[i,:,:d]-r.context.system.mean[i,:d])/r.context.system.scale[i,:d]).detach()
            truth=(cpu64(batch.unknowns.xi[i,:d])/r.context.system.scale[i,:d]).detach()
            values=torch.cat((z,truth[None]),0)
            im=axes[row,i].imshow(values,aspect='auto',cmap='RdBu_r');fig.colorbar(im,ax=axes[row,i],shrink=.7)
            axes[row,i].set(title=f'{name} {s.name}: standardized xi',xlabel='Lexicographic a, then b slots',yticks=range(5),yticklabels=['k=0','k=1','k=2','k=3','truth'])
    fig.suptitle(status+' — coefficient trajectories; deterministic estimates, not posterior samples');return fig


def sensor_iterations(results,batch,*,status):
    fig,axes=plt.subplots(len(results),3,figsize=(12,3*len(results)),squeeze=False,constrained_layout=True)
    for row,(name,r) in enumerate(results.items()):
        for i,s in enumerate(batch.specs):
            mask=r.context.system.valid[i];truth=r.context.system.y[i,mask].detach().numpy()
            ax=axes[row,i]
            for k in range(4):ax.scatter(truth,r.predictions[i,k,mask].detach().numpy(),s=10,alpha=.65,label=f'k={k}')
            if len(truth):ax.plot([truth.min(),truth.max()],[truth.min(),truth.max()],'k--')
            ax.set(title=f'{name} {s.name}',xlabel='Supplied noisy measurement',ylabel='Physical predicted measurement');ax.legend(fontsize=7)
    fig.suptitle(status+' — predicted versus supplied sensors, each iteration');return fig


def objective_histories(results,*,status):
    fig,axes=plt.subplots(1,3,figsize=(12,3),constrained_layout=True)
    for i,s in enumerate(next(iter(results.values())).context.specs):
        for name,r in results.items():axes[i].semilogy(range(4),r.objectives[i].detach().numpy(),'o-',label=name)
        axes[i].set(title=s.name,xlabel='Inverse iteration k',ylabel='Posterior objective Phi');axes[i].legend()
    fig.suptitle(status);return fig


def spatial_fields(model,result,*,status):
    item=result.spatial_history[-1];p=model.spatial.positions.detach().cpu();levels=model.spatial.levels.cpu()
    fig,axes=plt.subplots(2,4,figsize=(13,6),constrained_layout=True)
    for col,(i,q,label) in enumerate(((0,0,'Poisson source'),(1,0,'Heat initial u0'),(2,0,'Wave initial u0'),(2,1,'Wave initial v0'))):
        for row,(key,title) in enumerate((('fields','candidate field'),('projected_gradient','coefficient-gradient pushforward'))):
            values=item[key][i,:,q].detach().cpu();im=axes[row,col].scatter(p[:,0],p[:,1],c=values,cmap='RdBu_r',s=30)
            axes[row,col].scatter(p[levels==0,0],p[levels==0,1],marker='s',s=65,facecolors='none',edgecolors='black')
            axes[row,col].set(title=label+'\n'+title,xlim=(0,1),ylim=(0,1),aspect='equal');fig.colorbar(im,ax=axes[row,col],shrink=.7)
    fig.suptitle(status+' B1: k=2 memory inputs; source projection uses lambda*a / lambda*g_a, not a functional gradient');return fig


def spatial_attention(model,result,*,status):
    item=result.spatial_history[-1];p=model.spatial.positions.detach().cpu();levels=model.spatial.levels.cpu()
    fig,axes=plt.subplots(2,3,figsize=(12,7),constrained_layout=True)
    for i,s in enumerate(result.context.specs):
        im=axes[0,i].scatter(p[:,0],p[:,1],c=item['tokens'][i].detach().norm(dim=-1).cpu(),s=35,cmap='viridis')
        axes[0,i].scatter(p[levels==0,0],p[levels==0,1],marker='s',s=75,facecolors='none',edgecolors='black')
        axes[0,i].set(title=s.name+' memory feature norms',xlim=(0,1),ylim=(0,1),aspect='equal');fig.colorbar(im,ax=axes[0,i],shrink=.7)
        selected=[0,5,16,43,79];attention=item['attention'][-1][i].mean(0)[selected][:,result.context.observations.valid[i]].detach().cpu()
        axes[1,i].imshow(attention,aspect='auto',cmap='magma');axes[1,i].set(title='k=2 selected anchors, head mean',xlabel='Valid sensor index',yticks=range(5),yticklabels=selected)
    fig.suptitle(status+' — B1 attention is not proof of physical influence');return fig


def field_comparison(models,normalizers,*,status,resolution=64):
    dataset=CaseDataset(normalizers.root_seed);fig,axes=plt.subplots(4,1+2*len(models),figsize=(15,12),constrained_layout=True)
    for row,(name,quantity) in enumerate((('poisson',0),('heat',0),('wave',0),('wave',1))):
        case=dataset.case('train',name,7);spec=canonical_spec(name);obs=observations(dataset,case,normalizers,set_id='B-plot').inputs
        query=evaluation_grid(name,quantity_id=quantity,resolution=resolution,times=None if name=='poisson' else (.2,))
        truth=evaluate_batch((spec,),case_coefficients((case,)),query)[0,:,0].reshape(resolution,resolution).numpy();limit=max(abs(truth).max(),1e-9)
        items=[('truth',truth)]
        for approach,m in models.items():
            device=next(m.parameters()).device;dev=device.type if device.type in {'cpu','mps'} else str(device)
            with torch.no_grad():
                r=m.infer_unknowns(m.prepare_context((spec,),float32_batch(obs,device=dev)));pred=m.predict_queries(r,query).physical_value[0,:,0].reshape(resolution,resolution).numpy()
            items.extend([(approach,pred),(approach+' − truth',pred-truth)])
        for col,(label,value) in enumerate(items):
            im=axes[row,col].imshow(value,origin='lower',extent=(0,1,0,1),cmap='RdBu_r',vmin=-limit,vmax=limit);fig.colorbar(im,ax=axes[row,col],shrink=.7)
            axes[row,col].set_title(f"{name} {('u','v')[quantity]}: {label}",fontsize=9)
    fig.suptitle(status+' — physical fields, matched observations; dynamics at t=.2; no accuracy claim');return fig


def training_histories(reports):
    fig,axes=plt.subplots(1,2,figsize=(10,3.5),constrained_layout=True)
    for name,r in reports.items():
        axes[0].plot([v['update'] for v in r['updates']],[v['loss'] for v in r['updates']],'o-',label=name)
        axes[1].plot([v['update'] for v in r['updates']],[v['forward_seconds']+v['backward_seconds']+v['update_seconds'] for v in r['updates']],'o-',label=name)
    axes[0].set(xlabel='Successful optimizer update s',ylabel='Standardized coefficient MSE',title='Matched smoke inputs, no learning claim')
    axes[1].set(xlabel='Successful optimizer update s',ylabel='Forward + backward + optimizer seconds')
    for ax in axes:ax.legend()
    return fig

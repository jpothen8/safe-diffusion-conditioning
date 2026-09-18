"""Export scientific figures and a deterministic, non-reward-selected illustration."""
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation,PillowWriter
import core

def main():
    out=core.ROOT/'results/pendulum/confirmation'
    summary=json.loads((out/'summary.json').read_text())
    colors={'conditional':'#2878b5','projection':'#d87824','refinement':'#7554a3'}
    labels={'conditional':'Exact conditioning','projection':'Projection','refinement':'Finite refinement'}
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,3,figsize=(13.8,4.1),layout='constrained')
    for e,name in enumerate(['negative','positive']):
        result=summary['examples'][name]
        for j,method in enumerate(['conditional','refinement']):
            comparison=result[method+'_minus_projection'];x=e+(-.15 if j==0 else .15)
            mean=comparison['mean_difference'];lo,hi=comparison['family_ci98_75']
            axes[0].errorbar(x,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=colors[method],capsize=5,
                            label=labels[method] if e==0 else None)
        for j,method in enumerate(['conditional','projection','refinement']):
            metric=result['metrics'][method];x=e+(j-1)*.23
            axes[1].bar(x,100*metric['added_boundary_fraction'],width=.21,color=colors[method],
                        label=labels[method] if e==0 else None)
            axes[2].bar(x,metric['mean_capped_stabilization_time'],width=.21,color=colors[method])
    for ax in axes:ax.set_xticks([0,1],['Torque ≤ 0','Torque ≥ 0']);ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    axes[0].axhline(25,color='#555',ls='--',lw=1,label='Useful-effect threshold')
    axes[0].set_ylabel('Native return gain over projection');axes[0].set_ylim(0,75)
    axes[0].set_title('Held-out paired gains');axes[0].legend(fontsize=8,loc='lower left')
    axes[1].set_ylabel('Chunks touching zero-torque boundary (%)');axes[1].set_ylim(0,100)
    axes[1].set_title('Where projection puts probability');axes[1].legend(fontsize=8)
    axes[2].set_ylabel('Mean stabilization time (s)');axes[2].set_ylim(0,4.5)
    axes[2].set_title('All methods eventually stabilize')
    fig.suptitle('Pendulum: 0.6 s constrained chunk, common TD3 continuation, unchanged 10 s reward\n64 held-out states × 8 replicates; intervals adjusted for four comparisons',fontsize=12)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'comparison.{ext}',dpi=180)
    plt.close(fig)
    raw=np.load(out/'raw.npz')['chunk'];illustrations={};data={}
    fig,axes=plt.subplots(2,3,figsize=(10.5,6.6),layout='constrained')
    artists=[]
    for row,name in enumerate(['negative','positive']):
        # First coherent opposite-direction proposal, using torque alone, never reward.
        opposite=raw.mean((1,2))>1 if name=='negative' else raw.mean((1,2))<-1
        index=int(np.flatnonzero(opposite)[0]);illustrations[name]={'flat_index':index,'state_index':index//8,
            'replicate':index%8,'selection':'first raw proposal with mean torque in the opposite coherent mode; no reward ranking'}
        for col,method in enumerate(['conditional','projection','refinement']):
            roll=np.load(out/f'{name}_{method}.npz');data[(row,col)]=(roll,index)
            ax=axes[row,col];ax.set_aspect('equal');ax.set_xlim(-1.25,1.25);ax.set_ylim(-1.3,1.4)
            ax.set_xticks([]);ax.set_yticks([]);ax.spines[['left','bottom']].set_visible(False)
            ax.plot(0,0,'ko',ms=4);ax.plot([-.08,.08],[1.05,1.05],color='gray',lw=2)
            rod,=ax.plot([],[],lw=4,color=colors[method],marker='o',markevery=[1],ms=10)
            annotation=ax.text(0,-1.13,'',ha='center',fontsize=9)
            artists.append((rod,annotation,roll,index))
            native_return=float(roll['rewards'][index].sum())
            ax.set_title(f'{labels[method]}\nReturn {native_return:.1f}',fontsize=11)
            if col==0:ax.text(-1.25,.9,'Torque ≤ 0' if row==0 else 'Torque ≥ 0',rotation=90,ha='center',va='center')
    title=fig.suptitle('',fontsize=12)
    def update(frame):
        time=frame*.05
        title.set_text(f'Pendulum examples — t = {time:.1f} s\n'+('Torque restriction active' if frame<12 else 'Same unconstrained TD3 continuation'))
        for rod,annotation,roll,index in artists:
            theta=roll['states'][index,frame,0]
            rod.set_data([0,np.sin(theta)],[0,np.cos(theta)])
            annotation.set_text(f'Torque {float(roll["actions"][index,frame,0]):+.2f} Nm')
        return [title]+[a for row in artists for a in row[:2]]
    update(11);fig.savefig(out/'examples_at_0.55s.png',dpi=160)
    anim=FuncAnimation(fig,update,frames=range(0,200,2),interval=100,blit=False)
    anim.save(out/'examples.gif',writer=PillowWriter(fps=10),dpi=90)
    plt.close(fig);(out/'illustration_selection.json').write_text(json.dumps(illustrations,indent=2))
    print(json.dumps(illustrations,indent=2))

if __name__=='__main__':main()

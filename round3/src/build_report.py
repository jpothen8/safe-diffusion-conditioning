"""Generate tables and exportable figures directly from locked result summaries."""
import csv,json,os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR','/tmp/safety-conditioning-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]

def main():
    out=ROOT/'figures';out.mkdir(exist_ok=True);rows=[]
    for task in ['pendulum','Walker2d','HalfCheetah']:
        summary=json.loads((ROOT/f'results/{task}_comparison/summary.json').read_text())
        for cap,case in summary.get('cases',summary.get('examples')).items():
            bp='bayesfp_32_projected' if task=='pendulum' else 'bayesfp_projected'
            raw='bayesfp_32' if task=='pendulum' else 'bayesfp'
            q=case['comparisons']['conditional_minus_projection'];r=case['comparisons']['refinement_minus_'+bp]
            label=('Pendulum '+('negative' if cap=='negative' else 'positive')) if task=='pendulum' else task+' '+cap
            row={'task':task,'setting':cap,'label':label,'acceptance':case['acceptance'],
                 'q_minus_p':q['mean'],'q_ci':q['ci95'] if task=='pendulum' else q['ci99_375'],
                 'r_minus_bp':r['mean'],'r_ci':r['ci98_75'] if task=='pendulum' else r['ci99_375'],
                 'raw_bayes_unsafe':case['arms'][raw]['unsafe_outputs'],
                 'n':summary['config']['states']*summary['config']['replicates'],
                 'refinement_seconds':case['refinement_seconds'],
                 'bayes_seconds':case['particle_diagnostics']['32']['seconds'] if task=='pendulum' else case['bayesfp_seconds'],
                 'reference_bank_seconds_shared_caps':summary['sampling_seconds'],
                 'logical_proposals_mean':case['logical_proposals_mean'],
                 'q_vs_q_mmd':case['q_vs_q_mmd']}
            for label2,arm in [('q','conditional'),('p','projection'),('r','refinement'),('bp',bp)]:
                entry=case['arms'][arm]
                row[label2+'_return']=entry.get('mean_return',entry.get('return_mean'))
                row[label2+'_boundary']=entry.get('boundary_fraction',entry.get('added_boundary_fraction'))
                row[label2+'_mmd']=entry['mmd_to_conditional']
            rows.append(row)
    (ROOT/'results/table.json').write_text(json.dumps(rows,indent=2))
    with (ROOT/'results/table.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys());writer.writeheader();writer.writerows(rows)
    def interval(value,ci):return f'{value:+.2f} [{ci[0]:+.2f}, {ci[1]:+.2f}]'
    md=['| Setting | Conditioning − projection | Refinement − (BayesFP + projection) | Raw BayesFP unsafe |',
        '|---|---:|---:|---:|']
    for r in rows:md.append(f"| {r['label']} | {interval(r['q_minus_p'],r['q_ci'])} | {interval(r['r_minus_bp'],r['r_ci'])} | {r['raw_bayes_unsafe']}/{r['n']} |")
    md+=['','Intervals: conditioning–projection is descriptive 95% for Pendulum and planned 99.375% for MuJoCo. Refinement–BayesFP+projection is 98.75% for Pendulum and 99.375% for MuJoCo. They correct different predeclared comparison families. Native-return units differ by task. All hard-output arms had zero chunk violations and zero rejection refusals.']
    (ROOT/'results/TABLE.md').write_text('\n'.join(md)+'\n')
    tex=[r'\begin{table}[t]',r'\caption{Mean native return after one constrained chunk and identical continuation. Q: first-feasible rejection; P: exact projection; R: finite refinement; B+P: paper-derived BayesFP with 32 particles and exact terminal repair. Pendulum uses 64 states with 8 replicates and 10 s evaluation; MuJoCo uses 32 states with 4 replicates and 4 s evaluation.}',
         r'\label{tab:chunk-return}',r'\centering\small',r'\begin{tabular}{lrrrr}',r'\toprule',r'Setting & Q & P & R & B+P \\',r'\midrule']
    for r in rows:tex.append(r['label']+' & '+' & '.join(f'{r[k+"_return"]:.2f}' for k in ['q','p','r','bp'])+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}', '',r'\begin{table}[t]',
            r'\caption{Paired return differences with state-cluster bootstrap intervals. Q--P: 95\% (Pendulum, descriptive), 99.375\% (MuJoCo). R--(B+P): 98.75\% (Pendulum), 99.375\% (MuJoCo), adjusting their respective four- and eight-comparison families. Positive favors the first method. These are approximate bootstrap intervals, not finite-sample guarantees.}',
            r'\label{tab:chunk-differences}',r'\centering\footnotesize',r'\begin{tabular}{lrr}',r'\toprule',r'Setting & Q--P [interval] & R--(B+P) [interval] \\',r'\midrule']
    for r in rows:tex.append(r['label']+' & '+interval(r['q_minus_p'],r['q_ci'])+' & '+interval(r['r_minus_bp'],r['r_ci'])+r' \\')
    tex += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (ROOT/'paper_results.tex').write_text('\n'.join(tex)+'\n')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
    for ax,task,title in zip(axes,['pendulum','Walker2d','HalfCheetah'],['Pendulum','Walker2d','HalfCheetah']):
        rr=[r for r in rows if r['task']==task]
        for j,r in enumerate(rr):
            for key,ci,offset,color,marker,legend in [('q_minus_p','q_ci',.12,'#2166ac','o','Conditioning − projection'),('r_minus_bp','r_ci',-.12,'#b35806','s','Refinement − (BayesFP + projection)')]:
                value=r[key];bounds=r[ci]
                ax.errorbar(value,j+offset,xerr=[[value-bounds[0]],[bounds[1]-value]],color=color,fmt=marker,capsize=3,label=legend if j==0 else None)
        ax.axvline(0,color='.45',lw=1);ax.set_yticks(range(2),[r['setting'] for r in rr]);ax.invert_yaxis()
        ax.set_title(title);ax.set_xlabel('Native return difference');ax.grid(axis='x',alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncols=2)
    for ext in ['png','pdf','svg']:fig.savefig(out/f'return_comparisons.{ext}',dpi=180)
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained');y=np.arange(len(rows))
    for key,color,offset,label in [('p','#767676',-.23,'Projection'),('r','#b35806',0,'Refinement'),('bp','#008573',.23,'BayesFP + projection')]:
        axes[0].barh(y+offset,[100*r[key+'_boundary'] for r in rows],height=.21,color=color,label=label)
        axes[1].barh(y+offset,[r[key+'_mmd'] for r in rows],height=.21,color=color)
    axes[1].scatter([r['q_vs_q_mmd'] for r in rows],y,marker='|',s=180,color='black',label='Independent Q versus Q')
    for ax in axes:ax.set_yticks(y,[r['label'] for r in rows]);ax.invert_yaxis();ax.grid(axis='x',alpha=.15)
    axes[0].set_xlabel('Outputs touching the added boundary (%)');axes[1].set_xlabel('Action-kernel MMD to rejection (smaller is closer)')
    axes[0].legend(loc='lower right');axes[1].legend(loc='lower right')
    for ext in ['png','pdf','svg']:fig.savefig(out/f'distribution_diagnostics.{ext}',dpi=180)
    plt.close(fig)
    print('Generated six-setting tables, LaTeX tables and two PNG/PDF/SVG figures.')

if __name__=='__main__':main()

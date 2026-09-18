"""Independent selection, global box projection, replay and provenance audit."""
import hashlib,json
import numpy as np
import core

def main():
    root=core.ROOT;out=root/'results/pendulum/confirmation';old=root/'results/pendulum/confirmation_initial_precision'
    lock=json.loads((out/'lock.json').read_text())
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in lock['sha256'].items()}
    assert all(hashes.values()),hashes
    bank=np.load(out/'proposals.npz')['actions'];oldbank=np.load(old/'proposals.npz')['actions']
    np.testing.assert_array_equal(bank,oldbank)
    flat=bank.reshape(-1,64,12,1);raw=flat[:,0]
    assert np.isfinite(bank).all() and (bank>=-2).all() and (bank<=2).all()
    summary=json.loads((out/'summary.json').read_text());report={'locked_hashes_match':hashes,
        'all_32768_proposals_unchanged_by_precision_correction':True,'examples':{}}
    for name,lo,hi in [('negative',-2,0),('positive',0,2)]:
        masks=((flat>=lo)&(flat<=hi)).all((2,3));idx=masks.argmax(1)
        selection=np.load(out/f'{name}_selection.npz')
        np.testing.assert_array_equal(selection['acceptance'].reshape(masks.shape),masks)
        np.testing.assert_array_equal(selection['first_indices'].ravel(),idx)
        assert masks.any(1).all()
        q=np.load(out/f'{name}_conditional.npz')['chunk'];p=np.load(out/f'{name}_projection.npz')['chunk']
        np.testing.assert_array_equal(q,flat[np.arange(len(flat)),idx])
        np.testing.assert_array_equal(p,np.clip(raw,lo,hi))
        # Projection variational inequality, checked against every feasible proposal:
        # (raw - p) dot (candidate - p) <= 0 characterizes global Euclidean projection.
        inner=((raw-p)[:,None]*(flat-p[:,None])).sum((2,3))
        assert np.max(inner[masks])<=1e-10
        changes={};max_return_change=0.
        for method in ['conditional','projection','refinement']:
            data=np.load(out/f'{name}_{method}.npz');original=np.load(old/f'{name}_{method}.npz')
            np.testing.assert_array_equal(data['chunk'],original['chunk'])
            assert (data['chunk']>=lo).all() and (data['chunk']<=hi).all()
            assert np.isfinite(data['rewards']).all()
            max_return_change=max(max_return_change,float(np.max(np.abs(data['rewards'].sum(1)-original['rewards'].sum(1)))))
            changes[method]=True
        bystate=masks.reshape(64,8,64).mean((1,2))
        z=bystate;N=512.;zcrit=1.959963984540054
        center=(z+zcrit*zcrit/(2*N))/(1+zcrit*zcrit/N)
        half=zcrit*np.sqrt(z*(1-z)/N+zcrit*zcrit/(4*N*N))/(1+zcrit*zcrit/N)
        report['examples'][name]={'all_output_chunks_unchanged':changes,'all_outputs_feasible':True,
            'first_accepted_selection_exact':True,'global_euclidean_projection_exact':True,
            'acceptance_min_max':bystate[[bystate.argmin(),bystate.argmax()]].tolist(),
            'acceptance_wilson95_by_state':np.stack([center-half,center+half],1).tolist(),
            'logical_proposal_counts_quantiles':np.quantile(idx+1,[0,.5,.9,.99,1]).tolist(),
            'max_individual_return_change_from_precision_fix':max_return_change,
            'gym_max_replay_error':max(summary['examples'][name]['gym_max_error'].values())}
    means=bank.mean((3,4))
    report['shared_state_modes']={'negative_occupancy_range':[(means<-1).mean((1,2)).min(),(means<-1).mean((1,2)).max()],
        'positive_occupancy_range':[(means>1).mean((1,2)).min(),(means>1).mean((1,2)).max()]}
    report['passed']=True
    (out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['locked_hashes_match','examples']},indent=2))

if __name__=='__main__':main()

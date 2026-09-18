"""Development-only quality gate; no held-out confirmation results are used here."""
import json
import time
import numpy as np
import core
import pendulum as pd
import pendulum_train as dp

def closed_loop(model,states,interval,seed):
    states=states.copy();rs=[];aa=[];ss=[]
    for t in range(0,200,interval):
        chunks=dp.sample(model,pd.observation(states),seed+t)
        for j in range(min(interval,200-t)):
            ss.append(states.copy());aa.append(chunks[:,j])
            states,r=pd.dynamics(states,chunks[:,j]);rs.append(r)
    return dict(states=np.stack(ss,1),actions=np.stack(aa,1),rewards=np.stack(rs,1))

def main():
    start=time.perf_counter();out=core.ROOT/'results/pendulum/development';out.mkdir(exist_ok=True)
    model=dp.load();actor=pd.actor();rng=np.random.default_rng(63000)
    states=np.column_stack([np.pi+rng.uniform(-.05,.05,32),rng.uniform(-.05,.05,32)])
    repeated=np.repeat(states,64,axis=0)
    bank=dp.sample(model,pd.observation(repeated),63001)
    raw=pd.rollout(actor,repeated,bank)
    np.savez_compressed(out/'raw.npz',initial_states=states,proposals=bank.reshape(32,64,12,1),**raw)
    results={'phase':'development','seed':63000,'states':32,'proposals_per_state':64,
             'raw_mean_return':float(raw['rewards'].sum(1).mean()),
             'negative_mode':float((bank.mean((1,2)) < -1).mean()),
             'positive_mode':float((bank.mean((1,2)) > 1).mean()),
             'chunk_std_quantiles':np.quantile(bank.std(1),[0,.25,.5,.75,1]).tolist(),'constraints':{}}
    for name,lo,hi in [('negative',-2,0),('positive',0,2),('negative_relaxed',-2,1),('positive_relaxed',-1,2)]:
        feasible=((bank>=lo)&(bank<=hi)).all((1,2)).reshape(32,64)
        qidx=np.argmax(feasible,axis=1);valid=feasible.any(1)
        q=bank.reshape(32,64,12,1)[np.arange(32),qidx]
        projected=np.clip(bank.reshape(32,64,12,1)[:,0],lo,hi)
        brq=pd.rollout(actor,states,q);brp=pd.rollout(actor,states,projected)
        rq=brq['rewards'].sum(1);rp=brp['rewards'].sum(1)
        results['constraints'][name]={'acceptance':float(feasible.mean()),'refusals':int((~valid).sum()),
            'q_return':float(rq[valid].mean()),'p_return':float(rp[valid].mean()),
            'q_minus_p':float((rq-rp)[valid].mean()),'paired_differences':(rq-rp).tolist()}
        np.savez_compressed(out/(name+'.npz'),q=q,p=projected,feasible=feasible,q_indices=qidx,
                            q_states=brq['states'],p_states=brp['states'],q_rewards=brq['rewards'],p_rewards=brp['rewards'])
    # Test split was set by initial-state group before any overlapping chunk was made.
    data=np.load(core.ROOT/'results/pendulum/demonstrations.npz')
    test_states=data['initial_states'][data['test_groups']]
    results['baseline']={}
    for interval in [4,12]:
        roll=closed_loop(model,test_states,interval,64000+interval*100)
        returns=roll['rewards'].sum(1)
        np.savez_compressed(out/f'closed_loop_{interval}.npz',initial_states=test_states,**roll)
        results['baseline'][str(interval)]={'return_mean':float(returns.mean()),'returns':returns.tolist()}
    for reflection in [False,True]:
        roll=pd.rollout(actor,states,reflection=reflection)
        results['baseline']['expert_reflection_'+str(reflection)]={'return_mean':float(roll['rewards'].sum(1).mean()),
            'mean_first_chunk_torque':float(roll['actions'][:,:12].mean())}
    results['seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(results,indent=2));print(json.dumps(results,indent=2))

if __name__=='__main__':main()

"""Untouched modern MuJoCo public-expert smoke and shared-state behavior gate."""
import io,json,zipfile,time
from pathlib import Path
import numpy as np
import torch
import gymnasium as gym
from stable_baselines3 import SAC
ROOT=Path(__file__).resolve().parents[1]
torch.set_num_threads(2)

def load(task,level):
    path=ROOT/f'assets/{task}-v5-SAC-{level}.zip'
    with zipfile.ZipFile(path) as z:
        assert 'policy.pth' in z.namelist()
        metadata=json.loads(z.read('data'));sysinfo=z.read('system_info.txt').decode() if 'system_info.txt' in z.namelist() else ''
    model=SAC.load(path,device='cpu',custom_objects={'learning_rate':0.,'lr_schedule':lambda _:0.})
    return model,metadata,sysinfo

def rollout(task,model,seed,state=None,steps=500):
    env=gym.make(task+'-v5');obs,_=env.reset(seed=seed)
    if state is not None:
        env.unwrapped.set_state(np.asarray(state['qpos']),np.asarray(state['qvel']));obs=env.unwrapped._get_obs()
    actions=[];rewards=[];states=[];observations=[];infos=[];snapshots={}
    for t in range(steps):
        if t in [0,25,50,100]:snapshots[t]={'qpos':env.unwrapped.data.qpos.copy().tolist(),'qvel':env.unwrapped.data.qvel.copy().tolist()}
        observations.append(obs.copy());states.append(np.r_[env.unwrapped.data.qpos.copy(),env.unwrapped.data.qvel.copy()])
        action,_=model.predict(obs,deterministic=True);actions.append(action)
        obs,reward,terminated,truncated,info=env.step(action);rewards.append(reward);infos.append(info)
        if terminated or truncated:break
    result={'actions':np.array(actions),'observations':np.array(observations),'rewards':np.array(rewards),'states':np.array(states)}
    desc={'return':float(sum(rewards)),'steps':len(rewards),'dt':env.unwrapped.dt,
          'progress':float(infos[-1]['x_position']-states[0][0]),'mean_torque_rms':float(np.sqrt(np.mean(np.square(actions)))),
          'terminal':bool(terminated),'obs_dimension':env.observation_space.shape,'act_dimension':env.action_space.shape}
    env.close();return result,desc,snapshots

def main():
    out=ROOT/'results/locomotion_gate';out.mkdir(exist_ok=True)
    start=time.perf_counter();summary={}
    for task in ['Walker2d','HalfCheetah']:
        models={};summary[task]={'smoke':{},'shared_states':[]}
        for level in ['expert','medium']:
            model,metadata,sysinfo=load(task,level);models[level]=model
            (ROOT/f'records/{task}_{level}_saved_data.json').write_text(json.dumps(metadata,indent=2))
            (ROOT/f'records/{task}_{level}_system_info.txt').write_text(sysinfo)
            summary[task]['smoke'][level]=[]
            for seed in range(3):
                roll,desc,snaps=rollout(task,model,91000+seed,steps=1000)
                summary[task]['smoke'][level].append(desc)
                np.savez_compressed(out/f'{task}_{level}_smoke_{seed}.npz',**roll)
                if level=='expert':
                    for tick in [0,25,50,100]:
                        if tick in snaps:summary[task]['shared_states'].append({'seed':91000+seed,'tick':tick,'state':snaps[tick]})
        for i,row in enumerate(summary[task]['shared_states']):
            row['outcomes']={}
            for level,model in models.items():
                roll,desc,_=rollout(task,model,row['seed'],row['state'],500)
                row['outcomes'][level]=desc
                np.savez_compressed(out/f'{task}_shared_{i}_{level}.npz',**roll)
        print(json.dumps({task:summary[task]['smoke']}),flush=True)
    summary['seconds']=time.perf_counter()-start
    (out/'summary.json').write_text(json.dumps(summary,indent=2));print('Done',summary['seconds'],flush=True)

if __name__=='__main__':main()

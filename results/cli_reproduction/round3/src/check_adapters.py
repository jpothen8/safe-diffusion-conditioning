"""Check the BayesFP adapter against unchanged base sampling and autograd."""
import json
from pathlib import Path
import numpy as np
import torch
import bayesfp,chunk_model
ROOT=Path(__file__).resolve().parents[1]

def main():
    torch.set_num_threads(2);torch.manual_seed(110000)
    x=torch.randn((8,48),requires_grad=True)
    error=0.
    for kwargs in [{'radius':.72},{'lower':-float('inf'),'upper':0.}]:
        fun=bayesfp.ball_cost_gradient if 'radius' in kwargs else bayesfp.box_cost_gradient
        cost,analytic=fun(x,**kwargs);actual=torch.autograd.grad(cost.sum(),x,retain_graph=True)[0]
        error=max(error,float((actual-analytic).abs().max().detach()))
    assert error<1e-7
    uniform=bayesfp.systematic(torch.full((100,32),1/32))
    assert torch.equal(uniform,torch.arange(32)[None].expand(100,32))
    weights=torch.zeros((100,32));weights[:,7]=1
    assert (bayesfp.systematic(weights)==7).all()
    checks={'analytic_gradient_error':error,'uniform_resampling_preserves_population':True,'one_hot_resampling_correct':True}
    for task in ['Walker2d','HalfCheetah']:
        model=chunk_model.load(ROOT/f'assets/{task}_diffusion_v2.pt')
        snaps=json.loads((ROOT/f'results/{task}_comparison/snapshots.json').read_text())
        obs=np.array([s['observation'] for s in snaps[:4]])
        base=chunk_model.sample(model,obs,110001)
        adapted,diag=bayesfp.sample(model,obs,chunk_model.scheduler,model.dimension,110001,strength=0,particles=1)
        # The actual frozen sampler includes physical clipping after the DDPM.
        # Its last scheduler coefficient can overshoot 1 by about 1.4e-5.
        # The evaluation already performs this same clipping for every B arm.
        checks[task+'_unclipped_scheduler_overshoot']=float(np.maximum(np.abs(adapted)-1,0).max())
        assert np.array_equal(base,np.clip(adapted,-1,1))
        checks[task+'_zero_guidance_with_identical_postprocessing_bitwise_equal']=True
    checks['passed']=True
    (ROOT/'results/adapter_checks.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks))

if __name__=='__main__':main()

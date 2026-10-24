"""Paired finite/overflow/growth and optimizer/EMA trajectory equivalence gates."""
import argparse, gc, json
from pathlib import Path
import torch
from profile_train import Workload, write


def diff(xs, ys):
    sq=ref=0.; mx=0.
    for x,y in zip(xs,ys):
        d=x.detach().float()-y.detach().float()
        sq+=float(d.square().sum()); ref+=float(y.detach().float().square().sum())
        mx=max(mx,float(d.abs().max()))
    return {'relative_l2':(sq/max(ref,1e-30))**.5,'max_abs':mx}


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--weights',required=True)
    p.add_argument('--optimization',required=True);p.add_argument('--out',required=True)
    a=p.parse_args(); a.capture='none';a.batch=32;a.seed=240105
    candidate=a.optimization
    a.optimization='eager'; ref=Workload(a)
    a.optimization=candidate; fast=Workload(a)
    # Model initialization/graph warm-up must leave weights and EMA unchanged.
    initial=diff(fast.params,ref.params)
    assert initial['max_abs']==0
    ref.scaler.set_growth_interval(2);fast.scaler.set_growth_interval(2)
    rows=[]; gradient=None
    for step in range(20):
        batch=ref.inputs()
        # Explicit overflow exercises skipped optimizer + EMA and a changed scale.
        lr,nr=ref.update(batch,inject_inf=step==4)
        lf,nf=fast.update(batch,inject_inf=step==4)
        row={'step':step,'reference_loss':float(lr),'candidate_loss':float(lf),'reference_scale':ref.scaler.get_scale(),
             'candidate_scale':fast.scaler.get_scale(),'reference_skips':ref.skipped,'candidate_skips':fast.skipped}
        rows.append(row)
        assert row['reference_scale']==row['candidate_scale'] and ref.skipped==fast.skipped
        if step==0:gradient=diff([p.grad for p in fast.params],[p.grad for p in ref.params])
    checks={'initial_weights':initial,'first_clipped_gradient':gradient,'parameters':diff(fast.params,ref.params),
            'ema':diff(fast.avg,ref.avg)}
    for key in ('exp_avg','exp_avg_sq','step'):
        checks[key]=diff([fast.opt.state[p][key] for p in fast.params],[ref.opt.state[p][key] for p in ref.params])
    loss_rel=max(abs(r['candidate_loss']-r['reference_loss'])/abs(r['reference_loss']) for r in rows)
    limits={'first_gradient_relative_l2':.005,'parameter_relative_l2':.0001,'ema_relative_l2':.0001,'max_loss_relative':.005}
    passed=gradient['relative_l2']<limits['first_gradient_relative_l2'] and checks['parameters']['relative_l2']<limits['parameter_relative_l2'] and checks['ema']['relative_l2']<limits['ema_relative_l2'] and loss_rel<limits['max_loss_relative']
    result={'variant':a.variant,'optimization':candidate,'steps':20,'rows':rows,'checks':checks,'max_loss_relative':loss_rel,
            'limits_set_before_run':limits,'passed':passed,'scope':'20 full updates from trained EMA; same real batches; explicit Inf at step4; scaler growth interval2; FP32 parameters/states and FP16 compute unchanged'}
    write(a.out,result);print(json.dumps(result),flush=True)
    assert passed,'Numerical gate failed; retain result and do not promote this candidate'


if __name__=='__main__':main()

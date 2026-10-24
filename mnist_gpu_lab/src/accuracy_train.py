"""Accuracy follow-up: frozen validation search, then two predeclared test runs.

Keeps the original CNN family and CUDA Graph speedups. Tests are never read during
configuration search. A GPU scalar learning rate is mutated in place so the
captured optimizer actually sees cosine decay.
"""
import argparse, copy, gc, hashlib, json, math, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from train import CNN, evaluate, environment

CONFIGS=[
    dict(name='small_constant',filters=[8,16,32],dense=64,schedule='constant',augment=False),
    dict(name='small_cosine',filters=[8,16,32],dense=64,schedule='cosine',augment=False),
    dict(name='wide_cosine',filters=[16,32,64],dense=128,schedule='cosine',augment=False),
    dict(name='wide_aug_cosine',filters=[16,32,64],dense=128,schedule='cosine',augment=True),
]

def augment(x):
    # Each image gets independent small affine perturbations; no flips of digits.
    r=torch.rand((len(x),4),device=x.device)
    angle=(r[:,0]*2-1)*(10*math.pi/180)
    scale=.9+.2*r[:,1]
    c,s=angle.cos()/scale,angle.sin()/scale
    tx=(r[:,2]*2-1)*(4/28); ty=(r[:,3]*2-1)*(4/28)
    theta=torch.stack([c,-s,tx,s,c,ty],dim=1).reshape(-1,2,3)
    grid=F.affine_grid(theta,x.shape,align_corners=False)
    return F.grid_sample(x,grid,mode='bilinear',padding_mode='zeros',align_corners=False)

class AccuracyRunner:
    def __init__(self,model,x,y,batch=128,augmentation=False,graph=True):
        self.model,self.x,self.y,self.batch=model,x,y,batch
        self.augmentation=augmentation; self.graph=graph
        self.lr=torch.tensor(.001,device='cuda')
        self.opt=torch.optim.Adam(model.parameters(),lr=self.lr,eps=1e-7,fused=True,capturable=True)
        self.ids=torch.arange(batch,device='cuda'); self.metrics=torch.zeros(3,device='cuda')
        t=time.perf_counter()
        if graph:
            state=copy.deepcopy(model.state_dict())
            stream=torch.cuda.Stream(); stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(10): self.eager(self.ids)
            torch.cuda.current_stream().wait_stream(stream); torch.cuda.synchronize()
            self.g=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.g): self.eager(self.ids)
            self.model.load_state_dict(state)
            for state in self.opt.state.values():
                for tensor in state.values():
                    if torch.is_tensor(tensor): tensor.zero_()
            self.metrics.zero_(); torch.cuda.synchronize()
        self.setup_seconds=time.perf_counter()-t

    def eager(self,ids):
        self.opt.zero_grad(set_to_none=False)
        x=self.x[ids]; y=self.y[ids]
        if self.augmentation: x=augment(x)
        self.preview=x
        logits=self.model(x); loss=F.cross_entropy(logits,y)
        loss.backward(); self.opt.step()
        with torch.no_grad():
            self.metrics[0].add_(loss.detach()*len(ids))
            self.metrics[1].add_((logits.argmax(1)==y).sum())
            self.metrics[2].add_(len(ids))

    def step(self,ids):
        if self.graph and len(ids)==self.batch:
            self.ids.copy_(ids); self.g.replay()
        else: self.eager(ids)

def new_model(config,seed):
    torch.manual_seed(seed)
    return CNN(config['filters'],config['dense']).cuda()

def save_json(path,value):
    Path(path).write_text(json.dumps(value,indent=2))

def split_data(npz,out):
    images=npz['train_x']; labels=npz['train_y']; rng=np.random.default_rng(20261024)
    val=np.concatenate([rng.permutation(np.flatnonzero(labels==i))[:500] for i in range(10)])
    val.sort(); train=np.setdiff1d(np.arange(len(labels)),val)
    assert len(train)==55000 and len(val)==5000 and len(np.intersect1d(train,val))==0
    np.savez(out/'validation_split.npz',train_indices=train,val_indices=val)
    desc={'train_count':len(train),'validation_count':len(val),'per_class_validation':[500]*10,
          'seed':20261024,'validation_indices_sha256':hashlib.sha256(val.tobytes()).hexdigest(),
          'test_used_in_search':False}
    save_json(out/'split.json',desc)
    x=torch.from_numpy(images[:,None].copy()).cuda().float().div_(255)
    y=torch.from_numpy(labels.copy()).cuda()
    return x[torch.from_numpy(train).cuda()],y[torch.from_numpy(train).cuda()],x[torch.from_numpy(val).cuda()],y[torch.from_numpy(val).cuda()]

def check(x,y,out):
    cfg=CONFIGS[0]; a=new_model(cfg,42); b=new_model(cfg,42)
    ref=AccuracyRunner(a,x,y,graph=False); graph=AccuracyRunner(b,x,y)
    errors=[]
    for i,(lr,n) in enumerate([(.001,128),(.0002,17),(0.,128),(.00005,128)]):
        ids=torch.arange(i*128,i*128+n,device='cuda')
        ref.lr.fill_(lr); graph.lr.fill_(lr)
        before={k:v.clone() for k,v in b.state_dict().items()}
        ref.step(ids); graph.step(ids); torch.cuda.synchronize()
        errors.append(max((a.state_dict()[k]-b.state_dict()[k]).abs().max().item() for k in before))
        if lr==0: zero_lr_delta=max((before[k]-b.state_dict()[k]).abs().max().item() for k in before)
    aug=AccuracyRunner(new_model(cfg,42),x,y,augmentation=True)
    aug.step(torch.arange(128,device='cuda')); p1=aug.preview.clone()
    aug.step(torch.arange(128,device='cuda')); p2=aug.preview.clone(); torch.cuda.synchronize()
    rng_delta=(p1-p2).abs().mean().item()
    result={'changing_lr_weight_max_errors':errors,'zero_lr_parameter_delta':zero_lr_delta,
            'augmentation_replay_mean_pixel_difference':rng_delta,
            'passed':max(errors)<1e-5 and zero_lr_delta==0 and rng_delta>.001}
    save_json(out/'correctness.json',result); print('CHECK',json.dumps(result),flush=True)
    assert result['passed'],result

def train_one(config,seed,x,y,vx,vy,epochs,outdir,final=False):
    outdir.mkdir(parents=True,exist_ok=True)
    t0=time.perf_counter(); model=new_model(config,seed)
    runner=AccuracyRunner(model,x,y,augmentation=config['augment'])
    torch.cuda.reset_peak_memory_stats(); gen=torch.Generator().manual_seed(seed+1000)
    rows=[]; train_start=time.perf_counter()
    for epoch in range(epochs):
        lr=.001 if config['schedule']=='constant' else .00005+(.001-.00005)*(1+math.cos(math.pi*epoch/max(1,epochs-1)))/2
        runner.lr.fill_(lr); runner.metrics.zero_(); torch.cuda.synchronize(); t=time.perf_counter()
        order=torch.randperm(len(x),generator=gen).cuda()
        for ids in order.split(128): runner.step(ids)
        torch.cuda.synchronize(); training_s=time.perf_counter()-t
        loss_sum,correct,n=runner.metrics.cpu().tolist()
        assert n==len(x) and math.isfinite(loss_sum)
        row={'epoch':epoch+1,'lr':lr,'train_loss':loss_sum/n,'train_accuracy':correct/n,'train_seconds':training_s,'samples':int(n)}
        # Do not inspect the test set until a final model is completely trained.
        if not final:
            t=time.perf_counter(); metrics=evaluate(model,vx,vy); torch.cuda.synchronize()
            row.update(validation_accuracy=metrics['accuracy'],validation_loss=metrics['loss'],validation_seconds=time.perf_counter()-t)
        rows.append(row)
        save_json(outdir/'progress.json',{'config':config,'seed':seed,'epochs':rows})
        if epoch==0 or (epoch+1)%5==0: print(config['name'],seed,json.dumps(row),flush=True)
    torch.cuda.synchronize()
    result={'config':config,'seed':seed,'epochs':rows,'parameter_count':sum(p.numel() for p in model.parameters()),
            'train_seconds':sum(r['train_seconds'] for r in rows),'loop_wall_seconds':time.perf_counter()-train_start,
            'graph_setup_seconds':runner.setup_seconds,'training_pipeline_seconds':time.perf_counter()-t0,
            'peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,
            'train_count':len(x),'evaluation_split':'test_once_after_training' if final else 'validation'}
    if final:
        t=time.perf_counter(); evaluation=evaluate(model,vx,vy,return_probs=True); torch.cuda.synchronize()
        result['test_seconds']=time.perf_counter()-t
        probs=evaluation.pop('probabilities');np.save(outdir/'test_probabilities.npy',probs)
        result['test']=evaluation
    else: result['validation']={'accuracy':rows[-1]['validation_accuracy'],'loss':rows[-1]['validation_loss']}
    state={k:v.detach().cpu() for k,v in model.state_dict().items()}
    torch.save({'state_dict':state,'config':config,'seed':seed,'epochs':epochs},outdir/'model.pt')
    portable={'architecture':{'filters':config['filters'],'dense':config['dense'],'kernel':3,'pool':2,
                             'flatten':'NHWC','normalization':'uint8 / 255'},
              'tensors':{k:{'shape':list(v.shape),'data':v.flatten().tolist()} for k,v in state.items()}}
    (outdir/'model.json').write_text(json.dumps(portable,separators=(',',':')))
    if final:
        loaded=torch.load(outdir/'model.pt',map_location='cuda',weights_only=False)
        restored=new_model(config,seed);restored.load_state_dict(loaded['state_dict'])
        again=evaluate(restored,vx,vy,return_probs=True)['probabilities']
        result['export_max_probability_error']=float(np.max(np.abs(probs-again)))
        result['export_prediction_agreement']=float(np.mean(probs.argmax(1)==again.argmax(1)))
        del restored
    save_json(outdir/'metrics.json',result)
    print('DONE',config['name'],seed,json.dumps({k:result[k] for k in ['train_seconds','training_pipeline_seconds']}) ,result.get('test',result.get('validation')),flush=True)
    del runner,model;gc.collect();torch.cuda.empty_cache()
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['search','final']);p.add_argument('--epochs',type=int,default=20)
    p.add_argument('--data',default='data/mnist.npz');p.add_argument('--out',default='results/accuracy')
    args=p.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4);torch.set_num_interop_threads(1)
    torch.backends.cudnn.benchmark=True;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    npz=np.load(args.data)
    if args.action=='search':
        save_json(out/'environment_search.json',environment())
        save_json(out/'plan.json',{'candidates':CONFIGS,'epochs':args.epochs,'search_seed':42,
             'final_seeds':[42,43],'selection':'highest last-epoch validation accuracy; tie: lowest validation CE',
             'ensemble':'predeclared average of the two final seeds, no test-based weights or selection',
             'test_policy':'No test images/labels loaded during search; configuration frozen before final runs.'})
        x,y,vx,vy=split_data(npz,out);check(x,y,out)
        results=[train_one(c,42,x,y,vx,vy,args.epochs,out/c['name']) for c in CONFIGS]
        best=min(results,key=lambda r:(-r['validation']['accuracy'],r['validation']['loss']))
        selection={'config':best['config'],'epochs':args.epochs,'validation':best['validation'],'test_seen':False}
        save_json(out/'search_results.json',results);save_json(out/'selection.json',selection)
        print('SELECTED',json.dumps(selection),flush=True)
    else:
        selection=json.loads((out/'selection.json').read_text());save_json(out/'environment_final.json',environment())
        x=torch.from_numpy(npz['train_x'][:,None].copy()).cuda().float().div_(255);y=torch.from_numpy(npz['train_y'].copy()).cuda()
        tx=torch.from_numpy(npz['test_x'][:,None].copy()).cuda().float().div_(255);ty=torch.from_numpy(npz['test_y'].copy()).cuda()
        results=[train_one(selection['config'],seed,x,y,tx,ty,selection['epochs'],out/f'final_seed{seed}',True) for seed in [42,43]]
        a=np.load(out/'final_seed42/test_probabilities.npy');b=np.load(out/'final_seed43/test_probabilities.npy')
        ensemble=(a+b)/2;np.save(out/'ensemble_probabilities.npy',ensemble)
        labels=npz['test_y']; pred=ensemble.argmax(1)
        result={'selection':selection,'seeds':results,'ensemble':{'accuracy':float(np.mean(pred==labels)),
                 'errors':int(np.sum(pred!=labels)),'members':[42,43],'inference_model_count':2}}
        save_json(out/'final_results.json',result);print('FINAL',json.dumps({'single_model_test':[r['test']['accuracy'] for r in results],'ensemble':result['ensemble']}),flush=True)

if __name__=='__main__': main()

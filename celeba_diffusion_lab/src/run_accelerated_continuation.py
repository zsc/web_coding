"""Freeze source, copy a completed60k checkpoint, train/evaluate70k and80k."""
import argparse, datetime, hashlib, json, os, shutil, subprocess, sys
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--lr',type=float,required=True)
    a=p.parse_args();src=Path(a.source);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    prior=json.loads((src/'training.json').read_text())
    assert prior['step']==60000 and prior['status']=='complete' and prior['config']['finetune_lr']==a.lr
    assert not prior['config']['grayscale'] and prior['config']['batch']==32
    shutil.copy2(src/'checkpoint.pt',out/'checkpoint.pt');shutil.copy2(src/'training.json',out/'training.json')
    frozen=out/'source';shutil.copytree('src',frozen,ignore=shutil.ignore_patterns('__pycache__'))
    state={'source':str(src),'out':str(out),'start':60000,'target':80000,'milestones':[70000,80000],'lr':a.lr,
           'batch':32,'channels':3,'ppus':1,'acceleration':'graph','task_id':os.environ.get('SOFY_TASK_ID'),
           'source_checkpoint_sha256':hashlib.sha256((src/'checkpoint.pt').read_bytes()).hexdigest(),
           'source_hashes':{str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*.py')},
           'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commands':[],
           'scope':'Full60k model,optimizer,scaler,regular and slow EMA,RNG resume;RGB32;unchanged LR;70k/80k fixed KID evaluation on same reserved PPU.'}
    def save(status,**kw):
        state.update(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw)
        tmp=out/'pipeline.tmp';tmp.write_text(json.dumps(state,indent=2));tmp.replace(out/'pipeline.json')
        print(json.dumps({k:state[k] for k in ('status','task_id','updated_utc','start','target')}),flush=True)
    for step in state['milestones']:
        commands=[('training',[sys.executable,'-u',str(frozen/'train_finetune_fast.py'),'--variant','unet','--batch','32','--resume','--acceleration','graph','--finetune-lr',str(a.lr),'--steps',str(step),'--out',str(out)]),
                  ('evaluating',[sys.executable,'-u',str(frozen/'evaluate_accelerated_checkpoint.py'),'--root',str(out),'--step',str(step)])]
        for phase,cmd in commands:
            save(phase,current_milestone=step)
            logpath=out/f'{phase}_{step}.log'
            with logpath.open('w') as log:
                try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=2700).returncode
                except subprocess.TimeoutExpired:code=124
            state['commands'].append({'phase':phase,'step':step,'command':cmd,'exit_code':code,'log':str(logpath)})
            if code:
                save('failed');raise RuntimeError(f'{phase} returned{code}: {logpath}')
    save('complete')


if __name__=='__main__':main()

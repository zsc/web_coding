"""Freeze code, verify distributed numerics, then train with all16 PPUs."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--kind', choices=('unet','scaled'),required=True)
    p.add_argument('--source',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--lr',type=float,required=True)
    a=p.parse_args()
    out=Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    frozen=out/'source'
    shutil.copytree('src',frozen,ignore=shutil.ignore_patterns('__pycache__'))
    state={'config':vars(a),'task_id':os.environ.get('SOFY_TASK_ID'),'nodes':1,'ppus':16,'global_batch':512,
           'target_images':5120000,'created_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),
           'source_checkpoint_sha256':hashlib.sha256(Path(a.source).read_bytes()).hexdigest(),
           'source_hashes':{str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*.py')},'commands':[]}
    def save(status):
        state.update(status=status,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        tmp=out/'pipeline.tmp'
        tmp.write_text(json.dumps(state,indent=2))
        tmp.replace(out/'pipeline.json')
        print(json.dumps({'status':status,'task_id':state['task_id']}),flush=True)
    common=['--kind',a.kind,'--source',a.source,'--lr',str(a.lr),'--global-batch','512']
    for phase,script,extra in [('checking','check_distributed_acceleration.py',['--out',str(out/'numerical_gate.json')]),
                               ('training','train_distributed_fast.py',['--out',str(out)])]:
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nnodes=1','--nproc_per_node=16',str(frozen/script),*common,*extra]
        save(phase)
        logpath=out/f'{phase}.log'
        with logpath.open('w') as log:
            try:code=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=5400)
            except subprocess.TimeoutExpired:
                save('failed_timeout')
                raise
        state['commands'].append({'command':cmd,'exit_code':code.returncode,'log':str(logpath)})
        if code.returncode:
            save('failed')
            raise RuntimeError(f'{phase} failed:{code.returncode}; inspect {logpath}')
    save('complete')


if __name__=='__main__':main()

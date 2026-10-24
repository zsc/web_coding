"""Fresh-process, same-device AB/BA timing, plus numerical gates and ASight."""
import argparse, hashlib, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser();p.add_argument('--variant',required=True);p.add_argument('--weights',required=True);p.add_argument('--out',required=True)
    a=p.parse_args();out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    shutil.copytree('src',out/'source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    manifest={'commands':[],'task_id':os.environ.get('SOFY_TASK_ID'),'source_hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path('src').rglob('*.py')}}
    def run(name,cmd,limit=240):
        started=time.time()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:rc=proc.wait(timeout=limit)
            except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait();rc=124
        row={'name':name,'command':list(map(str,cmd)),'exit_code':rc,'seconds':time.time()-started}
        manifest['commands'].append(row);(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(row),flush=True)
        if rc:print((out/(name+'.log')).read_text()[-4000:],flush=True)
        return rc
    modes=['eager']
    for opt in (['graph'] if a.variant=='unet' else ['graph','triton','graph_triton']):
        cmd=[sys.executable,'-u','src/check_training_acceleration.py','--variant',a.variant,'--weights',a.weights,'--optimization',opt,'--out',out/(opt+'_check.json')]
        if run(opt+'_check',cmd)==0:modes.append(opt)
    base=[sys.executable,'-u','src/profile_train.py','--variant',a.variant,'--weights',a.weights]
    for rnd in range(5):
        for opt in modes if rnd%2==0 else modes[::-1]:
            assert run(f'r{rnd}_{opt}',base+['--optimization',opt,'--out',out/f'r{rnd}_{opt}','--seed',str(240105+rnd)])==0
    # Instrumented post-optimization trace is diagnostic only.
    asys='/usr/local/PPU_SDK/asight/bin/asys'
    for opt in ['eager']+[x for x in ('graph','graph_triton') if x in modes]:
        prefix=out/('asys_'+opt)
        rc=run('asys_'+opt,[asys,'profile','--trace','hggc,hgtx','--sample','none','--capture-range','hggcProfilerApi','--capture-range-end','stop','--max-report-size','128','--output',prefix]+base+['--optimization',opt,'--out',out/('trace_'+opt),'--capture','asys','--steps','2'])
        if not rc:run('export_'+opt,[asys,'export','--output',str(prefix)+'.sqlite',str(prefix)+'.asysrep'])
    manifest['complete']=True;manifest['accepted_modes']=modes
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))


if __name__=='__main__':main()

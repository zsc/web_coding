"""Bounded single-PPU profile suite; keep diagnostic and timing processes separate."""
import argparse, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path


def main():
    p=argparse.ArgumentParser(); p.add_argument('--variant',required=True); p.add_argument('--weights',required=True); p.add_argument('--out',required=True)
    a=p.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=True)
    manifest={'commands':[], 'task_id':os.environ.get('SOFY_TASK_ID'), 'source_hashes':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in Path('src').rglob('*.py')}}
    def run(name, cmd, limit=240):
        started=time.time()
        with (out/(name+'.log')).open('w') as log:
            proc=subprocess.Popen(list(map(str,cmd)),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try: code=proc.wait(timeout=limit)
            except subprocess.TimeoutExpired: os.killpg(proc.pid,signal.SIGKILL); proc.wait(); code=124
        row={'name':name,'command':list(map(str,cmd)),'exit_code':code,'seconds':time.time()-started}
        manifest['commands'].append(row); (out/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(row),flush=True)
        if code: print((out/(name+'.log')).read_text()[-4000:],flush=True)
        return code
    base=[sys.executable,'-u','src/profile_train.py','--variant',a.variant,'--weights',a.weights]
    asys='/usr/local/PPU_SDK/asight/bin/asys'
    assert run('timing',base+['--out',out/'timing'])==0
    assert run('torch',base+['--out',out/'torch','--capture','torch','--steps','2'])==0
    code=run('asys',[asys,'profile','--trace','hggc,hgtx,acdnn,acblas','--sample','none','--capture-range','hggcProfilerApi','--capture-range-end','stop','--max-report-size','128','--output',out/'asys']+base+['--out',out/'asys_workload','--capture','asys','--steps','2'])
    if not code: assert run('asys_export',[asys,'export','--output',out/'asys.sqlite',out/'asys.asysrep'])==0
    # Keep a frozen source snapshot beside the trace, without rewriting live trainers.
    import shutil
    shutil.copytree('src',out/'source',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)


if __name__=='__main__':main()

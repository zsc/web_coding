"""Single bounded PCM availability check, with raw result retained."""
import json, subprocess, time
from pathlib import Path

out=Path('results/ppu_acceleration/counter_probe');out.mkdir(parents=True,exist_ok=True)
cmd=['/usr/local/PPU_SDK/asight/bin/acu','--profile-from-start','no','--launch-count','1','--kill','no','--section','SpeedOfLight','--export',str(out/'probe'),'/usr/local/bin/python3','src/probe_asight_counter.py']
started=time.time()
try:
    r=subprocess.run(cmd,capture_output=True,text=True,timeout=60);code=r.returncode;log=r.stdout+'\n'+r.stderr
except subprocess.TimeoutExpired as e:
    code=124;log=str(e.stdout)+'\n'+str(e.stderr)
(out/'probe.log').write_text(log)
summary=('ASight Compute 单次探测未成功取得计数器；保留错误日志，未停止监控或更改驱动。瓶颈结论使用 ASight Systems 时间线。' if code else 'ASight Compute 探测命令成功；正式速度仍只取无 profiler 的计时，硬件计数器不用于外推整模型利用率。')
result={'command':cmd,'exit_code':code,'seconds':time.time()-started,'summary':summary,'files':[str(p) for p in out.iterdir()]}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);print(log[-4000:],flush=True)

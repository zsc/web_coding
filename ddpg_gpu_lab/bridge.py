"""Run SSH through the existing tmux server (macOS Terminal network context).

Usage: python3 bridge.py run 'remote shell command'
       python3 bridge.py push local_path remote_path
       python3 bridge.py pull remote_path local_path
"""
import pathlib, shlex, subprocess, sys, time, uuid
root = pathlib.Path(__file__).resolve().parent
jobs = root / '.bridge'
jobs.mkdir(exist_ok=True)
job = jobs / uuid.uuid4().hex
mode, *args = sys.argv[1:]
if mode == 'run':
    command = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', 'desk', args[0]]
elif mode == 'push':
    command = ['scp', '-r', args[0], 'desk:' + args[1]]
elif mode == 'pull':
    command = ['scp', '-r', 'desk:' + args[0], args[1]]
else:
    raise SystemExit('Expected run, push, or pull')
script = job.with_suffix('.sh')
log = job.with_suffix('.log')
status = job.with_suffix('.exit')
script.write_text('#!/bin/sh\n' + shlex.join(command) + ' > ' + shlex.quote(str(log)) + ' 2>&1\n' + 'echo $? > ' + shlex.quote(str(status)) + '\n')
subprocess.run(['tmux', 'new-window', '-d', '-n', 'ddpg-job', 'sh ' + shlex.quote(str(script))], check=True)
offset = 0
while True:
    if log.exists():
        with log.open() as f:
            f.seek(offset)
            print(f.read(), end='', flush=True)
            offset = f.tell()
    if status.exists():
        # Drain output written between the previous read and the exit sentinel.
        with log.open() as f:
            f.seek(offset)
            print(f.read(), end='', flush=True)
        sys.exit(int(status.read_text()))
    time.sleep(0.25)

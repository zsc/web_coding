"""Record source and checkpoint identity after completed affine measurements."""
import datetime,hashlib,importlib.metadata,json,platform,sys
from pathlib import Path
import torch
from model import gpu_status

def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()
files=[*Path('src').glob('*.py'),Path('run_affine_experiment.sh'),
       Path('data/manifest.json'),Path('data/gray_manifest.json')]
files += [Path('results')/v/'checkpoint.pt' for v in ('gray5k','time_affine5k','global_affine5k')]
files += [Path('results/affine_validation.json'),Path('results/affine_evaluation/distribution.json')]
r={'recorded_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'python':sys.version,
   'platform':platform.platform(),'cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
   'packages':{p:importlib.metadata.version(p) for p in ('torch','diffusers','numpy','pillow','matplotlib')},
   'gpu':gpu_status(),'files':[{'path':str(p),'bytes':p.stat().st_size,'sha256':digest(p)} for p in files]}
Path('results/affine_environment.json').write_text(json.dumps(r,indent=2));print('Recorded affine provenance.')

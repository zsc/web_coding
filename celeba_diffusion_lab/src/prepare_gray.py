"""One-channel luminance from the same decoded/cropped/resized RGB data."""
import json,time
from pathlib import Path
import numpy as np
start=time.perf_counter()
for split in ('train','valid','test'):
    rgb=np.load('data/'+split+'.npy',mmap_mode='r')
    gray=np.lib.format.open_memmap('data/'+split+'_gray.npy',mode='w+',dtype=np.uint8,shape=rgb.shape[:-1]+(1,))
    for i in range(0,len(rgb),512):
        chunk=np.array(rgb[i:i+512],dtype=np.float32)
        gray[i:i+512,:,:,0]=np.rint(chunk[:,:,:,0]*.299+chunk[:,:,:,1]*.587+chunk[:,:,:,2]*.114).astype(np.uint8)
    gray.flush();print(split,len(gray),flush=True)
Path('data/gray_manifest.json').write_text(json.dumps({'source':'same train/valid/test RGB uint8 caches',
    'transform':'one channel: round(0.299 R + 0.587 G + 0.114 B), applied after identical 64x64 preprocessing',
    'range':'uint8 0..255; normalized to [-1,1] for diffusion training','seconds':time.perf_counter()-start},indent=2))

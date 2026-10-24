"""Rebuild the exact grayscale cache from the existing NAS CelebA archive."""
import hashlib, io, json, time, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ARCHIVE='/dataset-cpfs5/vtm-foundation-pretrain/dataset_ori/celeba/archive (1).zip'
def init():
    global z,m
    z=zipfile.ZipFile(ARCHIVE);m={n.rsplit('/',1)[-1]:n for n in z.namelist() if n.endswith('.jpg')}
def convert(indices):
    rows=[]
    for i in indices:
        rgb=np.asarray(Image.open(io.BytesIO(z.read(m[f'{i:06d}.jpg']))).convert('RGB').crop((0,20,178,198)).resize((64,64),Image.Resampling.LANCZOS)).astype(np.float32)
        rows.append(np.rint(rgb[:,:,0]*.299+rgb[:,:,1]*.587+rgb[:,:,2]*.114).astype(np.uint8))
    return np.stack(rows)[...,None]
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
if __name__=='__main__':
    start=time.perf_counter();result={}
    # Official split membership is checked against the desk-generated filename lists.
    for split in ('train','valid'):
        indices=[int(n.split('.')[0]) for n in Path(f'data/{split}_filenames.txt').read_text().splitlines()]
        temp=Path(f'data/{split}_gray.nas.npy');a=np.lib.format.open_memmap(temp,mode='w+',dtype='uint8',shape=(len(indices),64,64,1))
        chunks=[indices[i:i+256] for i in range(0,len(indices),256)]
        with ProcessPoolExecutor(8,initializer=init) as pool:
            cursor=0
            for rows in pool.map(convert,chunks):a[cursor:cursor+len(rows)]=rows;cursor+=len(rows)
        a.flush();del a
        result[split]={'count':len(indices),'sha256':sha(temp),'seconds_total':time.perf_counter()-start}
        temp.replace(Path(f'data/{split}_gray.npy'));print(json.dumps({split:result[split]}),flush=True)
    Path('data/nas_cache_verification.json').write_text(json.dumps(result,indent=2))

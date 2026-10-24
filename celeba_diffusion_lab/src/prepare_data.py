"""Validate original filename coverage and preprocess directly from a CelebA ZIP."""
import argparse, hashlib, io, json, time, zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image

ARCHIVE=None
def init_worker(path):
    global ARCHIVE
    ARCHIVE=zipfile.ZipFile(path)

def decode_chunk(items):
    out=[]
    for filename, member, split, index in items:
        im=Image.open(io.BytesIO(ARCHIVE.read(member))).convert('RGB')
        if im.size!=(178,218): raise ValueError((filename,im.size))
        im=im.crop((0,20,178,198)).resize((64,64),Image.Resampling.LANCZOS)
        out.append((split,index,np.asarray(im)))
    return out

def digest(path,alg):
    h=hashlib.new(alg)
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser(); p.add_argument('--zip',default='data/Dataset.zip'); p.add_argument('--workers',type=int,default=8)
    args=p.parse_args(); start=time.perf_counter(); root=Path('data'); root.mkdir(exist_ok=True)
    partition=root/'list_eval_partition.txt'
    assert digest(partition,'md5')=='d32c9cbf5e040fd4025c592c306e6668'
    splitmap={name:int(split) for name,split in (s.split() for s in partition.read_text().splitlines())}
    assert len(splitmap)==202599
    with zipfile.ZipFile(args.zip) as z:
        images={Path(n).name:n for n in z.namelist() if n.lower().endswith('.jpg')}
    assert set(images)==set(splitmap),(len(images),list(set(splitmap)-set(images))[:5])
    counts=[sum(v==i for v in splitmap.values()) for i in range(3)]
    assert counts==[162770,19867,19962],counts
    names=['train','valid','test']; arrays=[np.lib.format.open_memmap(root/(n+'.npy'),mode='w+',dtype=np.uint8,shape=(c,64,64,3)) for n,c in zip(names,counts)]
    cursor=[0,0,0]; jobs=[]; files=[[],[],[]]
    for name,split in sorted(splitmap.items()):
        jobs.append((name,images[name],split,cursor[split])); files[split].append(name); cursor[split]+=1
    with ProcessPoolExecutor(args.workers,initializer=init_worker,initargs=(args.zip,)) as ex:
        chunks=[jobs[i:i+256] for i in range(0,len(jobs),256)]
        for k,result in enumerate(ex.map(decode_chunk,chunks)):
            for split,index,image in result: arrays[split][index]=image
            if k%100==0: print(json.dumps({'decoded':min((k+1)*256,len(jobs)),'seconds':time.perf_counter()-start}),flush=True)
    for a in arrays: a.flush()
    for n,fs in zip(names,files): (root/(n+'_filenames.txt')).write_text('\n'.join(fs)+'\n')
    from model import save_grid
    import torch
    # Unselected first 64 validation images: a reference panel, never used in training.
    save_grid(torch.from_numpy(np.array(arrays[1][:64])).permute(0,3,1,2).float()/127.5-1,'results/validation_reference.png')
    manifest={'archive':args.zip,'archive_sha256':digest(args.zip,'sha256'),'archive_md5':digest(args.zip,'md5'),
              'partition_md5':digest(partition,'md5'),'counts':dict(zip(names,counts)),
              'preprocessing':'RGB, original 178x218 -> center crop 178x178 at (0,20) -> Lanczos 64x64; stored uint8 NHWC',
              'filename_coverage':'all 202599 official filenames, no missing or extra JPGs',
              'source':'https://huggingface.co/datasets/student/celebA/tree/2d31e6555722815c74ea7c845b07c1063dd705e9',
              'official_archive_md5':'00d2c5bc6d35e252742224ab0c1e8fcb',
              'seconds':time.perf_counter()-start}
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest),flush=True)

if __name__=='__main__': main()

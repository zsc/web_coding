"""Lossless MNIST tile/IDX conversion; no resampling or captions."""
import argparse, hashlib, json, math, re, struct, time, urllib.request
from pathlib import Path
import numpy as np
from PIL import Image

REVISION = '0c253b32e07d973ba321916b9e19beb520e999b1'
BASE = f'https://raw.githubusercontent.com/zsc/zsc.github.io/{REVISION}/widgets/mnist/'
FILES = {
    'mnist_train_stitched.png': '2af1e950eff9ff3e5b1995722480562b5483ebc3',
    'mnist_test_stitched.png': 'd688d130a4d44686eeb78ca8a2f683fc33f211ce',
    'mnist_train_labels.js': '611e9990743dfc5c3c41958b5c19c94ddba0cf4a',
    'mnist_test_labels.js': '357d71506ee6458b7bb6763bf6f356efff9cdc65',
}

def unstitch(pixels, count, columns):
    h, w = pixels.shape
    assert w == columns * 28 and h % 28 == 0
    tiles = pixels.reshape(h//28, 28, columns, 28).transpose(0,2,1,3).reshape(-1,28,28)
    assert len(tiles) >= count
    return tiles[:count].copy()

def stitch(images, columns):
    rows = math.ceil(len(images)/columns)
    padded = np.zeros((rows*columns,28,28), dtype=np.uint8)
    padded[:len(images)] = images
    return padded.reshape(rows,columns,28,28).transpose(0,2,1,3).reshape(rows*28,columns*28)

def read_idx(path, images):
    raw = Path(path).read_bytes()
    if images:
        magic,n,h,w = struct.unpack('>4I',raw[:16])
        assert (magic,h,w) == (2051,28,28) and len(raw)==16+n*784
        return np.frombuffer(raw[16:],np.uint8).reshape(n,28,28).copy()
    magic,n = struct.unpack('>2I',raw[:8])
    assert magic==2049 and len(raw)==8+n
    return np.frombuffer(raw[8:],np.uint8).copy()

def prepare(root, idx_dir=None):
    root=Path(root); root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter(); manifest={'revision':REVISION,'files':{},'splits':{}}
    if idx_dir:
        for split,prefix,cols,variable in [('train','train',245,'mnistTrainLabels'),('test','t10k',100,'mnistTestLabels')]:
            images=read_idx(Path(idx_dir)/f'{prefix}-images-idx3-ubyte',True)
            labels=read_idx(Path(idx_dir)/f'{prefix}-labels-idx1-ubyte',False)
            assert len(images)==len(labels)
            Image.fromarray(stitch(images,cols)).save(root/f'mnist_{split}_stitched.png')
            (root/f'mnist_{split}_labels.js').write_text(f'var {variable} = '+json.dumps(labels.tolist())+';\n')
    else:
        for name,expected in FILES.items():
            path=root/name
            if not path.exists():
                print('Download',name,flush=True)
                with urllib.request.urlopen(BASE+name,timeout=90) as r: path.write_bytes(r.read())
            raw=path.read_bytes()
            git_sha=hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()
            assert git_sha==expected, f'Upstream hash mismatch: {name}'
    arrays={}
    for split,n,cols in [('train',60000,245),('test',10000,100)]:
        img=Image.open(root/f'mnist_{split}_stitched.png')
        pixels=np.asarray(img.convert('L'))
        labels_text=(root/f'mnist_{split}_labels.js').read_text()
        labels=np.asarray(json.loads(re.search(r'\[[\s\S]*\]',labels_text).group()),np.int64)
        assert len(labels)==n and labels.min()>=0 and labels.max()<10
        images=unstitch(pixels,n,cols)
        reconstructed=stitch(images,cols)
        assert np.array_equal(unstitch(reconstructed,n,cols),images)
        # Probe tile boundaries against direct rectangular crops, independent of reshape path.
        for i in [0,cols-1,cols,cols+1,n-1]:
            y,x=divmod(i,cols)
            assert np.array_equal(images[i],pixels[y*28:(y+1)*28,x*28:(x+1)*28])
        arrays[split+'_x']=images; arrays[split+'_y']=labels
        manifest['splits'][split]={'count':n,'width':img.width,'height':img.height,'mode':img.mode,
                                  'columns':cols,'empty_tiles':pixels.size//784-n,'roundtrip_exact':True}
    for path in root.iterdir():
        if path.name in FILES:
            manifest['files'][path.name]={'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}
    np.savez(root/'mnist.npz',**arrays)
    manifest['prepare_seconds']=time.perf_counter()-started
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',default='data'); p.add_argument('--idx-dir')
    a=p.parse_args(); prepare(a.data,a.idx_dir)

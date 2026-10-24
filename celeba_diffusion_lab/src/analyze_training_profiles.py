"""Summarize disjoint ASight kernel events and paired uninstrumented timings."""
import collections, hashlib, json, sqlite3, statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/ppu_acceleration'


def category(name):
    n=name.lower()
    if n.startswith('conv_'): return 'Convolution'
    if n.startswith('gemm_'): return 'GEMM'
    if 'flash_' in n: return 'Attention'
    if 'copy_' in n or 'copykernel' in n or 'copy_kernel' in n: return 'Copy / cast'
    if 'fusedadam' in n: return 'Fused AdamW'
    if 'groupnorm' in n or 'group_norm' in n or 'rowwisemoments' in n or 'computeinternalgradients' in n: return 'GroupNorm'
    if n.startswith('_forward') or n.startswith('_backward'): return 'Triton RMSNorm'
    if 'multi_tensor' in n: return 'Other multi-tensor'
    if 'reduce' in n: return 'Reduction'
    if 'elementwise' in n: return 'Elementwise'
    return 'Other'


def analyze(path):
    c=sqlite3.connect(path)
    rows=c.execute('SELECT k.start,k.end,s.value FROM HGPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON k.demangledName=s.id ORDER BY k.start').fetchall()
    assert len(rows)>500, 'Incomplete full-step trace'
    groups=collections.defaultdict(lambda: {'count':0,'total_ms':0.})
    names=collections.defaultdict(lambda: {'count':0,'total_ms':0.})
    spans=[]
    for a,b,n in rows:
        for d,key in ((groups,category(n)),(names,n)):
            d[key]['count']+=1; d[key]['total_ms']+=(b-a)/1e6
        if not spans or a>spans[-1][1]:spans.append([a,b])
        else:spans[-1][1]=max(spans[-1][1],b)
    span=(max(b for a,b,n in rows)-rows[0][0])/1e6
    union=sum(b-a for a,b in spans)/1e6
    return {'path':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'steps':2,'kernel_count':len(rows),'kernel_sum_ms':sum(b-a for a,b,n in rows)/1e6,
            'kernel_span_ms':span,'kernel_union_ms':union,'gap_within_kernel_span_ms':span-union,
            'groups':dict(sorted(groups.items(),key=lambda x:-x[1]['total_ms'])),
            'top_names':[{'name':n,**d} for n,d in sorted(names.items(),key=lambda x:-x[1]['total_ms'])[:30]],
            'caveat':'Instrumented two-step trace. Name-based groups are exclusive. Kernel union/span is timeline coverage, not occupancy or compute utilization. No hardware counter inference.'}


def copy_attribution():
    output={}
    for model in ('unet','large_jit'):
        path=ROOT/f'results/ppu_profile/{model}/torch/torch_trace.json'
        if not path.exists():continue
        events=json.loads(path.read_text())['traceEvents']
        cpu={e['args']['External id']:e for e in events if e.get('cat')=='cpu_op' and 'External id' in e.get('args',{})}
        groups=collections.defaultdict(lambda:{'count':0,'ms':0.}); examples={};unmatched=0
        for e in events:
            if e.get('cat')!='kernel' or 'copy' not in e.get('name','').lower():continue
            op=cpu.get(e.get('args',{}).get('External id'))
            if op is None:unmatched+=1;continue
            args=op.get('args',{});types=args.get('Input type',[]);strides=args.get('Input Strides',[]);shape=args.get('Input Dims',[])
            label='dtype conversion' if len(types)>1 and types[0]!=types[1] else 'layout conversion' if len(strides)>1 and strides[0]!=strides[1] else 'same dtype and stride / other copy'
            groups[label]['count']+=1;groups[label]['ms']+=e['dur']/1000
            key=str((label,shape[:2],types[:2],strides[:2]))
            row=examples.setdefault(key,{'kind':label,'shape':shape[:2],'types':types[:2],'strides':strides[:2],'count':0,'ms':0.})
            row['count']+=1;row['ms']+=e['dur']/1000
        output[model]={'groups':dict(groups),'unmatched_count':unmatched,'top_examples':sorted(examples.values(),key=lambda x:-x['ms'])[:15],
                       'scope':'Two-step torch trace; device copy kernels linked only via same-trace External id. Dtype changes take precedence if stride also differs; unmatched remain unmatched. No cross-profiler correlation IDs.'}
    (ROOT/'results/ppu_profile/copy_attribution.json').write_text(json.dumps(output,indent=2))


def main():
    copy_attribution()
    result={'models':{},'profiles':{},'protocol':'1 PPU per task; FP32 parameters/AdamW states; FP16 autocast; batch32 RGB64. Five fresh process pairs;20 warm-up steps and5x20 timed steps each; alternating order. Real CPU cache indexing, H2D and augmentation included. Excludes validation/checkpoint, initialization and graph setup. Same checkpoint and per-round seed. No profiler during timing.'}
    for kind,dirname in [('unet','unet'),('large_jit','large_jit_serial')]:
        d=OUT/dirname; modes={}
        for mode in ('eager','graph','triton','graph_triton'):
            runs=[]
            for i in range(5):
                p=d/f'r{i}_{mode}/result.json'
                if p.exists():runs.append(json.loads(p.read_text()))
            if runs:modes[mode]={'runs':runs,'n':len(runs),'median_ms':statistics.median(r['median_ms'] for r in runs),
                                 'peak_allocated_mib':max(r['peak_allocated_mib'] for r in runs),
                                 'measured_skips':sum(r['measured_skips'] for r in runs)}
        if 'eager' in modes:
            for mode,m in modes.items():
                ratios=[a['median_ms']/b['median_ms'] for a,b in zip(modes['eager']['runs'],m['runs'])]
                m['paired_speedup_median']=statistics.median(ratios)
                m['paired_speedup_range']=[min(ratios),max(ratios)]
        result['models'][kind]={'modes':modes,'checks':{p.stem:json.loads(p.read_text()) for p in d.glob('*_check.json')},
                                'complete':(json.loads((d/'manifest.json').read_text()).get('complete',False) if (d/'manifest.json').exists() else False)}
        for p in [d/f'asys_{mode}.sqlite' for mode in ('eager','graph','graph_triton') if (d/f'asys_{mode}.sqlite').exists()]:
            s=analyze(p); result['profiles'][kind+'_'+p.stem.removeprefix('asys_')]=s
    p=OUT/'resume_verification/verification.json'
    result['resume_verification']=json.loads(p.read_text()) if p.exists() else None
    result['failures']=[{'file':'results/ppu_profile/large_jit/asys.log','stage':'Initial ASight with hggc,hgtx,acdnn,acblas','status':'SIGSEGV; no trace, not counted as a successful profile'},
                        {'file':'results/ppu_acceleration/large_jit/r0_triton.log','stage':'First non-profiled Triton timing process','status':'SIGSEGV during another ASight job; cause unproven. Entire JiT matrix repeated after other profiler jobs finished; original evidence retained.'}]
    (OUT/'summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:{m: {'n':v['n'],'median_ms':v['median_ms'],'paired_speedup':v['paired_speedup_median']} for m,v in d['modes'].items()} for k,d in result['models'].items()},indent=2))


if __name__=='__main__':main()

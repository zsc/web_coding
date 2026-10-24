"""Generate a fully offline HTML report from actual machine-readable results."""
import base64, hashlib, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def read(path): return json.loads((ROOT/path).read_text())
def main():
    names=['cpu','cuda_eager','cuda_graph','cuda_graph_amp','default_subset','graph_10epochs',
           'cuda_eager_r1','cuda_graph_r1','default_subset_r1','cuda_eager_r2','cuda_graph_r2','default_subset_r2']
    runs={name:read(f'results/remote/{name}/metrics.json') for name in names if (ROOT/f'results/remote/{name}/metrics.json').exists()}
    d=np.load(ROOT/'data/mnist.npz'); probs=np.load(ROOT/'results/remote/cuda_graph/test_probabilities.npy')
    pred=probs.argmax(1); labels=d['test_y']; confusion=np.zeros((10,10),int)
    np.add.at(confusion,(labels,pred),1)
    payload={'runs':runs,'benchmark':read('results/remote/benchmark.json'),
             'browser':read('results/browser_baseline.json'),'correctness':read('results/remote/correctness.json'),
             'tfjs':read('results/tfjs_parity.json'),'manifest':read('data/manifest.json'),
             'provenance':read('original/provenance.json'),'labels':labels.tolist(),'predictions':pred.tolist(),
             'probabilities':np.round(probs.astype(float),7).tolist(),'confusion':confusion.tolist()}
    accuracy_dir=ROOT/'results/accuracy'
    if (accuracy_dir/'final_results.json').exists():
        improved=np.load(accuracy_dir/'final_seed42/test_probabilities.npy')
        improved_pred=improved.argmax(1); improved_confusion=np.zeros((10,10),int)
        np.add.at(improved_confusion,(labels,improved_pred),1)
        payload['accuracy']={'search':read('results/accuracy/search_results.json'),
                             'final':read('results/accuracy/final_results.json'),
                             'split':read('results/accuracy/split.json'),
                             'correctness':read('results/accuracy/correctness.json'),
                             'tfjs':read('results/accuracy/tfjs_parity.json'),
                             'predictions':improved_pred.tolist(),
                             'probabilities':np.round(improved.astype(float),7).tolist(),
                             'confusion':improved_confusion.tolist()}
    template=(ROOT/'src/report_template.html').read_text()
    if 'accuracy' in payload:
        template=template.replace('<section class="section" id="evidence">',
            (ROOT/'src/accuracy_section.html').read_text()+'\n<section class="section" id="evidence">',1)
        template=template.replace('</script></body></html>',
            (ROOT/'src/accuracy_report.js').read_text()+'\n</script></body></html>',1)
    encoded=json.dumps(payload,ensure_ascii=False,separators=(',',':')).replace('</','<\\/')
    png=base64.b64encode((ROOT/'data/mnist_test_stitched.png').read_bytes()).decode()
    report=template.replace('__REPORT_DATA__',encoded).replace('__TEST_PNG__',png)
    (ROOT/'report.html').write_text(report)
    (ROOT/'results/report_summary.json').write_text(json.dumps({
        'report_sha256':hashlib.sha256(report.encode()).hexdigest(),
        'run_names':list(runs),'errors':int((pred!=labels).sum()),'test_count':len(labels),
        'offline':True,'report_bytes':len(report.encode())},indent=2))
    print('Built',ROOT/'report.html',len(report.encode()),'bytes')

if __name__=='__main__': main()

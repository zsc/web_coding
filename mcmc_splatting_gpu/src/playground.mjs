// Inlined in report.html after its shared DATA, el(), paint() and download().
let liveTarget = Uint8Array.from(DATA.local.demo.target), worker = null, workerURL = null;
let latest = null, initializing = false, stopping = false;
paint('live-target', liveTarget);
paint('live-image', new Uint8Array(5000));

function setBusy(busy) {
  for (const id of ['demo-start', 'demo-text', 'demo-file', 'demo-N', 'demo-steps', 'demo-mode']) el(id).disabled = busy;
  el('demo-stop').disabled = !busy;
}
function disposeWorker() {
  worker?.terminate();
  worker = null;
  if (workerURL) URL.revokeObjectURL(workerURL);
  workerURL = null;
  initializing = stopping = false;
  setBusy(false);
}
function refresh(data) {
  latest = data;
  paint('live-image', data.image);
  el('live-iter').textContent = data.current.toLocaleString() + ' / ' + data.limit.toLocaleString();
  el('live-loss').textContent = data.loss.toFixed(1);
  el('live-accept').textContent = (100 * data.accepted / Math.max(1, data.current)).toFixed(1) + '%';
  el('live-time').textContent = (data.computeMs / 1000).toFixed(2) + ' s';
}
function exportSnapshot(data) {
  const snapshot = {width: 100, height: 50, iterations: data.current, loss: data.loss,
    mode: data.mode, gaussians: data.gaussians};
  download(new Blob([JSON.stringify(snapshot, null, 2)], {type: 'application/json'}), 'splats.json');
  const canvas = document.createElement('canvas');
  paint(canvas, data.image);
  canvas.toBlob(blob => download(blob, 'splats.png'));
}
function updateBudget() {
  const count = el('demo-N').valueAsNumber;
  el('demo-budget').textContent = Number.isFinite(count) && count >= 1 && count <= 10000
    ? `FP64 缓存约 ${(count * 5000 * 8 / 1048576).toFixed(1)} MiB；迭代数不增加缓存大小。`
    : '高斯数量范围：1–10,000。';
}
el('demo-N').addEventListener('input', updateBudget);
updateBudget();

el('demo-start').addEventListener('click', () => {
  for (const id of ['demo-N', 'demo-steps']) if (!el(id).reportValidity()) return;
  disposeWorker();
  latest = null;
  initializing = true;
  setBusy(true);
  el('demo-export').disabled = true;
  paint('live-image', new Uint8Array(5000));
  el('live-iter').textContent = '0 / ' + Number(el('demo-steps').value).toLocaleString();
  for (const id of ['live-loss', 'live-accept', 'live-time']) el(id).textContent = '—';
  el('live-status').textContent = '正在后台初始化缓存，可随时停止…';
  try {
    workerURL = URL.createObjectURL(new Blob([PLAYGROUND_WORKER_SOURCE], {type: 'text/javascript'}));
    const active = worker = new Worker(workerURL);
    active.onmessage = ({data}) => {
      if (worker !== active) return;
      if (data.type === 'error') {
        disposeWorker();
        el('demo-export').disabled = true;
        el('live-status').textContent = '计算失败：' + data.message;
        return;
      }
      if (data.type === 'export') { exportSnapshot(data); return; }
      initializing = false;
      refresh(data);
      el('demo-export').disabled = false;
      if (data.type === 'done' || data.type === 'stopped') {
        disposeWorker();
        el('live-status').textContent = data.type === 'done'
          ? '运行完成，可下载当前结果或重新初始化。'
          : '已停止，可下载当前状态；再次运行会重新初始化。';
      } else if (!stopping) {
        el('live-status').textContent = `本机 CPU Worker 计算中 · ${(100 * data.current / data.limit).toFixed(2)}% · 可随时停止`;
      }
    };
    active.onerror = error => {
      disposeWorker();
      el('demo-export').disabled = true;
      el('live-status').textContent = '无法运行后台计算：' + error.message;
    };
    active.postMessage({type: 'start', count: Number(el('demo-N').value), limit: Number(el('demo-steps').value),
      mode: el('demo-mode').value, target: liveTarget});
  } catch (error) {
    disposeWorker();
    el('live-status').textContent = '无法启动后台计算：' + error.message;
  }
});
el('demo-stop').addEventListener('click', () => {
  if (initializing) {
    disposeWorker();
    el('live-status').textContent = '已取消初始化，可以调整参数重新运行。';
  } else if (worker) {
    stopping = true;
    el('demo-stop').disabled = true;
    el('live-status').textContent = '正在停止，保留当前结果…';
    worker.postMessage({type: 'stop'});
  }
});
function resetTarget(a) {
  disposeWorker();
  liveTarget = a;
  latest = null;
  paint('live-target', a);
  paint('live-image', new Uint8Array(5000));
  el('demo-export').disabled = true;
  el('live-iter').textContent = '0';
  for (const id of ['live-loss', 'live-accept', 'live-time']) el(id).textContent = '—';
  el('live-status').textContent = '目标已更新，点击初始化并运行。';
}
el('demo-text').addEventListener('change', () => {
  el('demo-file').value = '';
  if (el('demo-text').value === 'MCMC') { resetTarget(Uint8Array.from(DATA.local.demo.target)); return; }
  const c = document.createElement('canvas');
  c.width = 100; c.height = 50;
  const x = c.getContext('2d');
  x.fillStyle = 'black'; x.fillRect(0, 0, 100, 50);
  x.fillStyle = 'white'; x.font = 'bold 25px sans-serif'; x.textAlign = 'center'; x.textBaseline = 'middle';
  x.fillText(el('demo-text').value, 50, 25, 96);
  const a = x.getImageData(0, 0, 100, 50).data;
  resetTarget(Uint8Array.from({length: 5000}, (_, i) => a[i * 4]));
});
el('demo-file').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  setBusy(true);
  el('demo-stop').disabled = true;
  try {
    const bm = await createImageBitmap(file), c = document.createElement('canvas');
    c.width = 100; c.height = 50;
    const x = c.getContext('2d');
    x.fillStyle = 'black'; x.fillRect(0, 0, 100, 50); x.drawImage(bm, 0, 0, 100, 50); bm.close();
    const a = x.getImageData(0, 0, 100, 50).data;
    resetTarget(Uint8Array.from({length: 5000}, (_, i) => Math.round(.2126 * a[4 * i] + .7152 * a[4 * i + 1] + .0722 * a[4 * i + 2])));
  } catch (err) { el('live-status').textContent = '无法读取图片：' + err.message; }
  finally { setBusy(false); }
});
el('target-export').addEventListener('click', () => download(new Blob(['P5\n100 50\n255\n', liveTarget], {type: 'image/x-portable-graymap'}), 'target.pgm'));
el('demo-export').addEventListener('click', () => {
  if (worker) worker.postMessage({type: 'export'});
  else if (latest?.gaussians) exportSnapshot(latest);
});

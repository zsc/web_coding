import {CachedChain, initialize, propose, rng} from './core.mjs';

// Embedded with core.mjs in a Blob Worker by build_report.py, including for file://.
let chain, random, current = 0, limit = 0, computeMs = 0, mode, timer, lastUpdate = 0;
let running = false;

function snapshot(type, includeParameters = false) {
  const image = chain.image();
  self.postMessage({type, current, limit, computeMs, mode, loss: chain.loss,
    accepted: chain.accepted, image, ...(includeParameters ? {gaussians: chain.gs} : {})}, [image.buffer]);
}

function tick() {
  if (!running) return;
  try {
    const begin = performance.now();
    while (current < limit && performance.now() - begin < 12) {
      const temp = mode === 'fixed' ? 50 : 50 * Math.pow(.05 / 50, current / Math.max(1, limit - 1));
      chain.step(propose(chain.gs, 100, 50, random), temp);
      current++;
      if (current % 5000 === 0) chain.rebuild();
    }
    computeMs += performance.now() - begin;
    if (current === limit) {
      running = false;
      snapshot('done', true);
    } else {
      if (performance.now() - lastUpdate >= 100) {
        snapshot('progress');
        lastUpdate = performance.now();
      }
      timer = setTimeout(tick, 0);
    }
  } catch (error) {
    running = false;
    self.postMessage({type: 'error', message: error.message});
  }
}

self.onmessage = ({data}) => {
  try {
    if (data.type === 'start') {
      if (!Number.isInteger(data.count) || data.count < 1 || data.count > 10000 ||
          !Number.isInteger(data.limit) || data.limit < 1 || data.limit > 10000000) {
        throw new Error('高斯数范围为 1–10,000，迭代数范围为 1–10,000,000。');
      }
      chain = new CachedChain(initialize(100, 50, data.count, rng(42)), 100, 50, data.target);
      random = rng(101);
      limit = data.limit;
      mode = data.mode;
      current = computeMs = 0;
      running = true;
      snapshot('ready');
      timer = setTimeout(tick, 0);
    } else if (data.type === 'stop' && chain) {
      running = false;
      clearTimeout(timer);
      snapshot('stopped', true);
    } else if (data.type === 'export' && chain) {
      snapshot('export', true);
    }
  } catch (error) {
    running = false;
    self.postMessage({type: 'error', message: error.message});
  }
};

// Exercise the unmodified upstream training functions (only asset URLs are localized).
const fs = require('node:fs');
const {chromium}=require('/Users/zsc/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
(async()=>{
  const browser=await chromium.launch({headless:true,executablePath:'/Applications/Google Chrome.app/Contents/MacOS/Google Chrome'});
  const page=await browser.newPage({viewport:{width:1280,height:900}});
  page.on('console',m=>{if(/Epoch|backend|Error/.test(m.text()))console.log(m.text())});
  page.on('pageerror',e=>console.error(e.message));
  const start=Date.now();
  await page.goto('http://127.0.0.1:8767/original_local.html');
  await page.waitForFunction(()=>typeof trainDataLoaded!=='undefined' && trainDataLoaded && testDataLoaded,{},{timeout:120000});
  const load_seconds=(Date.now()-start)/1000;
  const result=await page.evaluate(async()=>{
    const gl=document.createElement('canvas').getContext('webgl');
    const ext=gl?.getExtension('WEBGL_debug_renderer_info');
    const environment={user_agent:navigator.userAgent,tfjs:tf.version.tfjs,backend:tf.getBackend(),
      renderer:ext?gl.getParameter(ext.UNMASKED_RENDERER_WEBGL):null};
    const runs=[];
    for(let i=0;i<3;i++){
      if(model) model.dispose();
      const t=performance.now();await trainModel();
      runs.push({seconds:(performance.now()-t)/1000,
        train_accuracy:[...accuracyChart.data.datasets[0].data],test_accuracy:[...accuracyChart.data.datasets[1].data]});
    }
    return {environment,runs,config:{train_n:6000,test_n:1000,epochs:5,batch:128,filters:[8,16,32],dense:64,lr:.001},
      note:'Original trainModel; only asset URLs localized and TF.js pinned to 4.22.0. Run 1 includes first-use shader compilation; runs 2–3 reuse browser kernels. New random model initialization each run.'};
  });
  result.load_seconds=load_seconds;
  fs.writeFileSync('results/browser_baseline.json',JSON.stringify(result,null,2));
  console.log(JSON.stringify(result));await browser.close();
})().catch(e=>{console.error(e);process.exit(1)});

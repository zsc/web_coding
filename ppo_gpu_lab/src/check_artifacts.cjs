const fs=require('fs'),path=require('path'),vm=require('vm');
const root=path.resolve(__dirname,'..');
for(const file of ['webgpu.html','report.html','play.html']){
 const html=fs.readFileSync(path.join(root,file),'utf8'),ids=[...html.matchAll(/\bid="([^"]+)"/g)].map(m=>m[1]);
 // Inline JS has HTML strings with ids, so duplicate-id checks use HTML parsing in the browser.
 let code=0,json=0,worker=0;
 for(const m of html.matchAll(/<script([^>]*)>([\s\S]*?)<\/script>/g)){
  if(m[1].includes('application/json')){JSON.parse(m[2]);json++;}
  else {new vm.Script(m[2],{filename:file});if(m[1].includes('text/plain'))worker++;else code++;}
 }
 if(/__(REPORT|LIVE|WEBGPU)_\w+__/.test(html))throw Error('Unresolved placeholder '+file);
 console.log(`${file}: ${code} scripts, ${worker} Worker, ${json} JSON payloads valid`);
}
for(const seed of [71,79,83]){
 const r=JSON.parse(fs.readFileSync(path.join(root,`results/webgpu_train_seed${seed}.json`)));
 if(r.completedIterations!==1200||r.timing.length!==1200||r.best.flat.length!==819||!r.best.flat.every(Number.isFinite))throw Error('Invalid result '+seed);
}
console.log('Three complete training results: 1200 timings and 819 finite weights each');

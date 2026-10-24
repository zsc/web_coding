export async function validateWebGPU(e,fixture,model){
 const d=e.device,mb=e.config.minibatch,B=e.b,N=e.config.n,T=e.config.horizon;
 if(mb!==2048)throw Error('Validation fixture requires minibatch 2048');
 const maxdiff=(a,b)=>Math.max(...a.map((v,i)=>Math.abs(v-b[i])));
 const checks={};
 async function run(commands){const enc=d.createCommandEncoder(),pass=enc.beginComputePass();for(const [name,count,offset=0]of commands)e.dispatch(pass,name,count,offset);pass.end();d.queue.submit([enc.finish()]);await d.queue.onSubmittedWorkDone();if(e.fatal)throw Error(e.fatal);}
 e.config.entropy=.01;e.writeConfig(1);d.queue.writeBuffer(e.w,0,new Float32Array(fixture.initial));d.queue.writeBuffer(e.opt,0,new Float32Array(1638));
 const r=new Float32Array(B*16+16),idx=new Uint32Array(B*e.config.epochs),up=new Uint32Array(e.updates*64);
 for(let i=0;i<mb;i++){r.set(fixture.obs[i],i*16);r[i*16+6]=fixture.raw[i];r[i*16+7]=fixture.oldlp[i];r[i*16+13]=fixture.adv[i];r[i*16+14]=fixture.ret[i];idx[i]=i;}
 for(let i=0;i<20;i++){up[i*64]=0;up[i*64+1]=i+1;}d.queue.writeBuffer(e.roll,0,r);d.queue.writeBuffer(e.indices,0,idx);d.queue.writeBuffer(e.upd,0,up);
 await run([['forward_backward',mb/64],['gradients',819]]);let g=Array.from(await e.read(e.grad,819));
 checks.gradientMaxAbs=maxdiff(g,fixture.gradient);checks.gradientRelativeL2=Math.sqrt(g.reduce((s,v,i)=>s+(v-fixture.gradient[i])**2,0)/fixture.gradient.reduce((s,v)=>s+v*v,0));
 await run([['adam',13]]);checks.adam1MaxAbs=maxdiff(Array.from(await e.read(e.w,819)),fixture.after1);
 const cmds=[];for(let i=1;i<20;i++)cmds.push(['forward_backward',mb/64,i*256],['gradients',819,i*256],['adam',13,i*256]);await run(cmds);
 checks.adam20MaxAbs=maxdiff(Array.from(await e.read(e.w,819)),fixture.after20);
 let seed=19;const rand=()=>{seed=(Math.imul(1664525,seed)+1013904223)>>>0;return(seed+.5)/4294967296};
 r.fill(0);const expected=[];
 for(let i=0;i<mb;i++){const s=Array.from({length:6},()=>rand()*2-1),a=rand()*2-1;r.set(s,i*16);r[i*16+6]=a;expected.push(model.step(Array.from(r.slice(i*16,i*16+6)),Math.fround(a)*e.config.force,e.config.force));}
 d.queue.writeBuffer(e.roll,0,r);await run([['test_physics',mb/64]]);const phys=await e.read(e.act,6*mb);let physError=0;for(let i=0;i<mb;i++)for(let j=0;j<6;j++)physError=Math.max(physError,Math.abs(phys[j*mb+i]-expected[i][j]));checks.physicsMaxAbs=physError;
 r.fill(0);for(let i=0;i<B;i++){const o=i*16;r[o+8]=(rand()-.5)*5;r[o+9]=(rand()-.5)*2;r[o+10]=+(rand()<.08);r[o+11]=+(r[o+10]||rand()<.1);r[o+12]=(rand()-.5)*5;}
 const advantage=new Float64Array(B),returns=new Float64Array(B);
 for(let n=0;n<N;n++){let last=0;for(let t=T-1;t>=0;t--){const i=t*N+n,o=i*16;last=r[o+9]+.99*r[o+12]*(1-r[o+10])-r[o+8]+.99*.95*(1-r[o+11])*last;advantage[i]=last;returns[i]=last+r[o+8];}}
 const mean=advantage.reduce((a,b)=>a+b,0)/B,std=Math.sqrt(advantage.reduce((a,b)=>a+(b-mean)**2,0)/B)+1e-8;
 d.queue.writeBuffer(e.roll,0,r);await run([['gae',Math.ceil(N/64)],['stats_partial',B/256],['stats_final',1],['normalize',B/256]]);const gae=await e.read(e.roll,B*16+16);
 let retError=0,advError=0;for(let i=0;i<B;i++){retError=Math.max(retError,Math.abs(gae[i*16+14]-returns[i]));advError=Math.max(advError,Math.abs(gae[i*16+13]-(advantage[i]-mean)/std));}checks.gaeReturnsMaxAbs=retError;checks.normalizedAdvantagesMaxAbs=advError;
 // Check stochastic impulses inside and across rollout blocks. A nonterminal
 // stored next value must equal the next observation's value, before any update.
 e.reset(fixture.initial,173);e.writeConfig(1);
 await run([['rollout',Math.ceil(N/32)]]);const block1=await e.read(e.roll,B*16);
 await run([['rollout',Math.ceil(N/32)]]);const block2=await e.read(e.roll,B*16);
 let continuity=0,transitions=0,impulses=0;
 for(let t=0;t<T;t++)for(let n=0;n<N;n++){
  const o=(t*N+n)*16;if(block1[o+11])continue;
  const next=t+1<T?block1:block2,ni=((t+1)%T*N+n)*16;
  continuity=Math.max(continuity,Math.abs(block1[o+12]-next[ni+8]));transitions++;
  const ns=model.step(Array.from(block1.slice(o,o+6)),Math.max(-1,Math.min(1,block1[o+6]))*e.config.force,e.config.force);
  if(Math.abs(ns[1]-next[ni+1])>1e-4)impulses++;
 }
 checks.rolloutValueContinuityMaxAbs=continuity;
 const thresholds={gradientMaxAbs:2e-5,gradientRelativeL2:1e-4,adam1MaxAbs:2e-5,adam20MaxAbs:1e-4,physicsMaxAbs:1e-5,gaeReturnsMaxAbs:2e-5,normalizedAdvantagesMaxAbs:2e-5,rolloutValueContinuityMaxAbs:1e-6};
 return {passed:impulses>0&&Object.entries(checks).every(([k,v])=>Number.isFinite(v)&&v<thresholds[k]),checks,thresholds,rolloutTransitions:transitions,observedImpulses:impulses,reference:fixture.reference,samples:mb,gaeSamples:B,device:e.info(),note:'FP32 approximate agreement; not bitwise identity. Physics checked against independent scalar Float64 LDLT. Includes true termination, time truncation, stochastic impulses and rollout-block continuity.'};
}

import {WebGPUPPO} from './webgpu_ppo.mjs';
import {validateWebGPU} from './webgpu_validate.mjs';
import {assessPolicy,assessSuite} from './webgpu_eval.mjs';
import './live_model.js';
let engine=null,initial=null,fixture=null,busy=false,stopRequested=false,target=1200;
let history=[],timing=[],best=null,baseline=null,activeMs=0,phase='idle';
const send=(type,data={})=>postMessage({type,...data});
const median=a=>{const x=a.slice().sort((a,b)=>a-b);return(x[Math.floor((x.length-1)/2)]+x[Math.floor(x.length/2)])/2;};
const quantile=(a,q)=>a.slice().sort((a,b)=>a-b)[Math.floor((a.length-1)*q)];
async function checkpoint(final=false){
 const current=await engine.snapshot();const evaluation=assessPolicy(current.flat,engine.config.force,PendulumModel,{count:128,seed:2026,degrees:10});
 const measure={iteration:engine.iteration,evaluation,meanReward:current.meanReward,terminals:current.terminals};history.push(measure);
 if(!best||evaluation.successRate>best.evaluation.successRate||(evaluation.successRate===best.evaluation.successRate&&evaluation.meanReturn>best.evaluation.meanReturn))best={...current,evaluation};
 send('progress',{iteration:engine.iteration,target,timing:timing.at(-1),history,trainingMs:timing.reduce((a,x)=>a+x.wallMs,0),bestIteration:best.iteration,bestSuccess:best.evaluation.successRate,current});
 return current;
}
async function train(){
 busy=true;stopRequested=false;phase='training';const started=performance.now();send('state',{state:'training',target});
 try{
  while(engine.iteration<target&&!stopRequested){
   const t=await engine.step(engine.config.robust?Math.min(1,engine.iteration/200):0);timing.push(t);
   if(engine.iteration%50===0||engine.iteration===target)await checkpoint();
   else if(engine.iteration%10===0)send('progress',{iteration:engine.iteration,target,timing:t,trainingMs:timing.reduce((a,x)=>a+x.wallMs,0),history,bestIteration:best?.iteration,bestSuccess:best?.evaluation.successRate});
  }
  if(stopRequested){const current=await engine.snapshot();send('paused',{current,iteration:engine.iteration,target});}
  else{
   phase='evaluating';send('state',{state:'evaluating'});
   const latest=await engine.snapshot(),heldOut=assessSuite(best.flat,engine.config.force,PendulumModel);
   activeMs+=performance.now()-started;
   const result={device:engine.info(),target,completedIterations:engine.iteration,initialization:'Prior CUDA PPO seed 7, same raw weights; output force uses selected 40 N or 10 N. No LQR / teacher / policy fallback.',training:{warmup:'No warmup excluded from training time',trainingMs:timing.reduce((a,x)=>a+x.wallMs,0),gpuMs:timing.reduce((a,x)=>a+(x.gpuMs||0),0),activeMs,setupMs:engine.setupMs,samples:engine.iteration*engine.b},history,timing,baseline,heldOut,best,latest};
   send('complete',{result});return;
  }
  activeMs+=performance.now()-started;
 }finally{busy=false;phase='idle';}
}
self.onmessage=async({data:m})=>{
 if(m.type==='pause'){if(phase==='training'){stopRequested=true;send('state',{state:'pausing'});}return;}
 if(busy){send('error',{message:'当前操作尚未结束。'});return;}
 try{
  if(m.type==='init'){
   busy=true;send('state',{state:'compiling'});engine?.destroy();initial=m.initial;fixture=m.fixture;
   engine=await WebGPUPPO.create(initial,m.config);history=[];timing=[];best=null;baseline=null;activeMs=0;target=m.target||1200;
   baseline=assessSuite(initial,engine.config.force,PendulumModel);await checkpoint();busy=false;send('ready',{device:engine.info(),baseline,target});
  }else if(m.type==='train'){target=m.target||target;await train();}
  else if(m.type==='validate'){
   busy=true;send('state',{state:'validating'});const check=await WebGPUPPO.create(initial,{...engine.config,force:40,robust:true});
   try{send('validation',{result:await validateWebGPU(check,fixture,PendulumModel)});}finally{check.destroy();busy=false;}
  }else if(m.type==='benchmark'){
   busy=true;send('state',{state:'benchmarking'});const bench=await WebGPUPPO.create(initial,{n:256,horizon:64,minibatch:2048,epochs:10,seed:917,force:10,robust:false});
   try{for(let i=0;i<5;i++)await bench.step(0);const records=[];for(let i=0;i<30;i++)records.push(await bench.step(0));
    const wall=records.map(x=>x.wallMs),gpu=records.map(x=>x.gpuMs).filter(Number.isFinite);send('benchmark',{result:{device:bench.info(),workload:'Physical original task, force 10N, threshold 12deg, reward original, no disturbances; 256x64 / minibatch 2048 / 10 epochs / FP32 / 819 parameters',warmup:5,repeats:30,medianWallMs:median(wall),p10WallMs:quantile(wall,.1),p90WallMs:quantile(wall,.9),medianGpuMs:gpu.length?median(gpu):null,sps:16384/(median(wall)/1000),records,timingBoundary:'Wall includes Fisher-Yates, buffer uploads, encoding, submission and GPU completion. Excludes query map, metrics, evaluation, compilation, UI. GPU timestamp spans all training kernels.'}});
   }finally{bench.destroy();busy=false;}
  }else if(m.type==='snapshot'){send('snapshot',{current:await engine.snapshot()});}
 }catch(e){busy=false;phase='idle';send('error',{message:e.stack||e.message});}
};

const fs = require('fs');
const os = require('os');
const core = require('./core_original.cjs');
const opts = Object.fromEntries(process.argv.slice(2).map(x=>x.replace(/^--/,'').split('=')));
let rngState = +(opts.seed || 1);
Math.random = () => { rngState |= 0; rngState = rngState + 0x6D2B79F5 | 0; let t = Math.imul(rngState ^ rngState >>> 15, 1 | rngState); t ^= t + Math.imul(t ^ t >>> 7, 61 | t); return ((t ^ t >>> 14) >>> 0) / 4294967296; };
const config = {STATE_DIM:4,ACTION_DIM:1,HIDDEN_DIM1:32,HIDDEN_DIM2:16,ACTION_BOUND:10,LR_ACTOR:0.0001,LR_CRITIC:0.001,GAMMA:0.99,TAU:0.005,BUFFER_SIZE:50000,BATCH_SIZE:64,NOISE_STDDEV:0.2};
const envConfig = {gravity:9.8,masscart:1,masspole:0.1,length:0.5,tau:0.02,action_bound:10,theta_threshold_radians:12*Math.PI/180,x_threshold:2.4};
core.configure(config);
const agent = new core.DDPGAgent(config);
const env = new core.InvertedPendulumEnv(envConfig);
function evaluate() {
  let savedRng=rngState; rngState=20261024;
  let lengths=[],returns=[],trajectory=[];
  for(let i=0;i<64;i++) {
    const e = new core.InvertedPendulumEnv(envConfig); let s=[0,0,((i*37)%64+0.5)/64*0.02-0.01,((i*17)%64+0.5)/64*0.02-0.01], r=0, n=0; e.state=s.slice();
    for(let j=0;j<200;j++) { const a=agent.getAction(s,false); const t=e.step(a); if(i===0) trajectory.push({s:s.slice(),a:a[0]}); s=t.nextState; r+=t.reward; n++; if(t.done)break; }
    lengths.push(n); returns.push(r);
  }
  rngState=savedRng;
  return {mean_length:lengths.reduce((a,b)=>a+b)/64,success200:lengths.filter(x=>x===200).length/64,mean_return:returns.reduce((a,b)=>a+b)/64,trajectory};
}
function populate(n=5000) {let s=env.reset(),ep=0;for(let i=0;i<n;i++){let a=[(Math.random()*2-1)*10];let t=env.step(a);agent.replayBuffer.add(s,a,t.reward,t.nextState,t.done);s=t.nextState;if(t.done||++ep===200){s=env.reset();ep=0;}}}
const mode=opts.mode||'bench'; let out;
if(mode==='fixture') {
  populate(256); const batch=agent.replayBuffer.sample(64);
  const before={actor:agent.actor.getWeights(),critic:agent.critic.getWeights(),target_actor:agent.targetActor.getWeights(),target_critic:agent.targetCritic.getWeights()};
  const predictions=batch.map(e=>({a:agent.actor.predict(e.s),q:agent.critic.predict(e.s,e.a),dqda:agent.critic.get_dQ_da(e.s,e.a)}));
  const physics=batch.map(e=>{env.state=e.s.slice();return env.step(e.a);});
  agent.replayBuffer.sample=()=>batch;
  const losses=agent.learn();
  out={config,envConfig,batch,before,predictions,physics,losses,after:{actor:agent.actor.getWeights(),critic:agent.critic.getWeights(),target_actor:agent.targetActor.getWeights(),target_critic:agent.targetCritic.getWeights()}};
} else if(mode==='bench') {
  populate(); for(let i=0;i<100;i++)agent.learn();
  const rounds=[]; const updates=+(opts.updates||300);
  for(let r=0;r<5;r++){let t=performance.now();for(let i=0;i<updates;i++)agent.learn();rounds.push((performance.now()-t)/updates);}
  out={mode:'original_node_learn',config,node:process.version,host:os.hostname(),cpu:os.cpus()[0].model,rounds_ms:rounds,median_ms:[...rounds].sort((a,b)=>a-b)[2],updates_per_round:updates,scope:'learn includes replay sample, 64 serial critic SGD updates, actor update and Polyak; no DOM'};
} else if(mode==='train') {
  let history=[],s=env.reset(),ep=0;let steps=+(opts.steps||20000),start=performance.now();
  for(let i=1;i<=steps;i++){const a=agent.getAction(s);const t=env.step(a);agent.replayBuffer.add(s,a,t.reward,t.nextState,t.done);if(agent.replayBuffer.size()>=64)agent.learn();s=t.nextState;if(t.done||++ep===200){s=env.reset();ep=0;}if(i%2000===0){const e=evaluate();delete e.trajectory;history.push({steps:i,seconds:(performance.now()-start)/1000,...e});console.error(JSON.stringify(history.at(-1)));}}
  out={mode:'original_node_train',config,seed:+(opts.seed||1),steps,seconds:(performance.now()-start)/1000,history,evaluation:evaluate(),actor:agent.actor.getWeights()};
} else throw Error('Unknown mode');
fs.writeFileSync(opts.out||'results/baseline.json',JSON.stringify(out,null,2));
console.log(JSON.stringify({mode,path:opts.out,median_ms:out.median_ms,seconds:out.seconds,evaluation:out.evaluation&&{...out.evaluation,trajectory:undefined}}));

const fs = require('fs'), {performance} = require('perf_hooks');
const root = require('path').resolve(__dirname, '..');
let seed = 20261024;
Math.random = () => {seed=(Math.imul(1664525,seed)+1013904223)>>>0; return (seed+0.5)/4294967296;};
const {DoubleInvertedPendulumEnv:Env, PPOAgent, Memory} = require('./repaired_core.cjs');
const makeAgent=()=>new PPOAgent(6,1,16,3e-4,1e-3,.99,.95,.2,10,64,.01);
let original_error='';
try {const {NeuralNetwork}=require('./original_core.cjs'); const n=new NeuralNetwork(6,[16,16],2);n.forward([.1,.2,.01,.3,.02,.1]);const g=n.backward([1,1]);if(g.weights.flat(Infinity).some(x=>!Number.isFinite(x)))original_error='Non-finite gradients: Mat.map callback receives (value,column,rowArray), not (value,row,column)';}catch(e){original_error=e.toString();}
const env=new Env(), agent=makeAgent(), fixture={physics:[],actor:agent.actor.getWeights(),critic:agent.critic.getWeights(),forward:[]};
for(let i=0;i<256;i++) {const state=Array.from({length:6},(_,j)=>(Math.random()-.5)*([4,2,.38,2,.38,2][j]));const action=Math.random()*2-1;env.state=[...state];const result=env.step(action);fixture.physics.push({state,action,...result});}
for(const f of fixture.physics.slice(0,16))fixture.forward.push({state:f.state,actor:agent.actor.forward(f.state),critic:agent.critic.forward(f.state)});
const updateAgent=makeAgent();updateAgent.ppoEpochs=1;
fixture.update={actor:JSON.parse(JSON.stringify(updateAgent.actor.getWeights())),critic:JSON.parse(JSON.stringify(updateAgent.critic.getWeights()))};
const mem=new Memory();let state=env.reset();
for(let i=0;i<64;i++){const a=updateAgent.selectAction(state),r=env.step(a.action);mem.add(state,a.action,r.reward,r.done,a.log_prob,a.value);state=r.done?env.reset():r.observation;}
fixture.update.data=JSON.parse(JSON.stringify(mem.get()));
fixture.update.losses=updateAgent.train(mem);
fixture.update.actor_after=updateAgent.actor.getWeights();fixture.update.critic_after=updateAgent.critic.getWeights();
fs.writeFileSync(root+'/results/fixtures.json',JSON.stringify(fixture));
const result={host:require('os').hostname(),node:process.version,original_error,config:{steps:2048,epochs:10,minibatch:64,hidden:16},runs:[]};
for(let run=0;run<4;run++) {
  let s=env.reset(), len=0, m=new Memory();let t=performance.now();
  for(let i=0;i<2048;i++){const a=agent.selectAction(s),r=env.step(a.action);m.add(s,a.action,r.reward,r.done,a.log_prob,a.value);s=r.observation;len++;if(r.done||len>=1000){s=env.reset();len=0;}}
  const rollout_ms=performance.now()-t;t=performance.now();const losses=agent.train(m);const update_ms=performance.now()-t;
  result.runs.push({rollout_ms,update_ms,total_ms:rollout_ms+update_ms,...losses});
}
console.log(JSON.stringify(result,null,2));
fs.writeFileSync(root+'/results/js_baseline.json',JSON.stringify(result,null,2));

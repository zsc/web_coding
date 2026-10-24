// Compare one actual GPU Adam update with independent full-image finite differences.
import fs from 'node:fs';
import {hybridInitial} from './hybrid_webgpu.mjs';
const actual=JSON.parse(fs.readFileSync('results/webgpu_hybrid_one_step.json')),target=JSON.parse(fs.readFileSync('assets/zhou_target.json')).pixels;
const W=320,H=128,P=W*H,init=hybridInitial(target,W,H,32,1),sum=new Float64Array(P);
function contribution(g,p){const dx=p%W-g[0],dy=Math.floor(p/W)-g[1],co=Math.cos(-g[4]),si=Math.sin(-g[4]),u=(dx*co-dy*si)/g[2],v=(dx*si+dy*co)/g[3];return g[5]*Math.exp(-.5*(u*u+v*v));}
for(let j=0;j<32;j++){const g=Array.from(init.slice(j*12,j*12+6));for(let p=0;p<P;p++)sum[p]+=contribution(g,p);}
let maxError=0,checked=0;const failures=[],nearZero=[];
for(let j=0;j<32;j++){
 const g=Array.from(init.slice(j*12,j*12+6)),base=Float64Array.from({length:P},(_,p)=>contribution(g,p));
 for(let k=0;k<6;k++){
  const eps=1e-4,plus=[...g],minus=[...g];if(k===2||k===3){plus[k]*=Math.exp(eps);minus[k]*=Math.exp(-eps);}else{plus[k]+=eps;minus[k]-=eps;}
  let diff=0;for(let p=0;p<P;p++){const r=sum[p]-base[p]-target[p],a=r+contribution(plus,p),b=r+contribution(minus,p);diff+=(a*a-b*b);}
  const grad=diff/(2*eps);if(Math.abs(grad)<1e-5){nearZero.push({gaussian:j,parameter:k,gradient:grad,observed_delta:actual.gaussians[j*12+k]-g[k]});continue;}const rate=[.15,.15,.025,.025,.025,1][k];let expected=(k===2||k===3?Math.log(g[k]):g[k])-rate*grad/(Math.abs(grad)+1e-8);
  if(k===2||k===3)expected=Math.min(32,Math.max(.1,Math.exp(expected)));if(k===0)expected=Math.max(0,Math.min(W,expected));if(k===1)expected=Math.max(0,Math.min(H,expected));if(k===5)expected=Math.max(0,Math.min(255,expected));
  const error=Math.abs(actual.gaussians[j*12+k]-expected);maxError=Math.max(maxError,error);checked++;if(error>3e-5)failures.push({j,k,error,grad});
 }
}
const result={method:'GPU one Adam step vs independent Float64 central finite differences of the full smooth image objective',gaussians:32,parameters_checked:checked,near_zero_gradient_threshold:1e-5,near_zero_not_compared:nearZero,near_zero_reason:'Adam first step normalizes by gradient magnitude; numerical noise near zero makes the finite-difference update ill-conditioned, especially with ROI truncation and fixed-point rendering.',maximum_absolute_parameter_error:maxError,tolerance:3e-5,failures};
fs.writeFileSync('results/webgpu_hybrid_gradient_check.json',JSON.stringify(result,null,2)+'\n');console.log(result);if(failures.length)process.exitCode=1;

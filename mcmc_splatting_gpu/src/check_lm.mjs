// Independent Float64 central differences for the actual CUDA normal equations.
import fs from 'node:fs';
import assert from 'node:assert/strict';
const root=new URL('../',import.meta.url);
const audit=JSON.parse(fs.readFileSync(new URL('results/desk_lm/audit.json',root)));
const target=JSON.parse(fs.readFileSync(new URL('assets/zhou_target.json',root))).pixels;
assert.equal(audit.chains,1);
const P=audit.W*audit.H;
function contribution(g){const [x,y,sx,sy,a,amp]=g,co=Math.cos(-a),si=Math.sin(-a),out=new Float64Array(P);
 for(let py=0,p=0;py<audit.H;py++)for(let px=0;px<audit.W;px++,p++){const dx=px-x,dy=py-y,u=(dx*co-dy*si)/sx,v=(dx*si+dy*co)/sy;out[p]=amp*Math.exp(-.5*(u*u+v*v));}return out;}
const sum=new Float64Array(P);for(const g of audit.initial){const a=contribution(g);for(let p=0;p<P;p++)sum[p]+=a[p];}
const residual=sum.map((v,p)=>v-target[p]);
function rel(a,b){return Math.hypot(...a.map((v,i)=>v-b[i]))/Math.max(1e-12,Math.hypot(...a));}
function solve(H,b){const L=Array.from({length:6},()=>Array(6).fill(0));for(let i=0;i<6;i++)for(let j=0;j<=i;j++){let s=H[i][j];for(let k=0;k<j;k++)s-=L[i][k]*L[j][k];L[i][j]=i===j?Math.sqrt(s):s/L[j][j];}
 const y=[],x=Array(6).fill(0);for(let i=0;i<6;i++){let s=b[i];for(let k=0;k<i;k++)s-=L[i][k]*y[k];y[i]=s/L[i][i];}for(let i=5;i>=0;i--){let s=y[i];for(let k=i+1;k<6;k++)s-=L[k][i]*x[k];x[i]=s/L[i][i];}return x;}
const checks=[];
for(let j=0;j<audit.N;j++){
 const g=audit.initial[j],J=[];
 for(let k=0;k<6;k++){const eps=1e-4,plus=[...g],minus=[...g];if(k===2||k===3){plus[k]*=Math.exp(eps);minus[k]*=Math.exp(-eps);}else{plus[k]+=eps;minus[k]-=eps;}
  const a=contribution(plus),b=contribution(minus);J.push(a.map((v,p)=>(v-b[p])/(2*eps)));}
 const grad=Array(6).fill(0),H=Array.from({length:6},()=>Array(6).fill(0)),upper=[];
 for(let k=0;k<6;k++){for(let p=0;p<P;p++)grad[k]+=J[k][p]*residual[p];for(let l=k;l<6;l++){let s=0;for(let p=0;p<P;p++)s+=J[k][p]*J[l][p];H[k][l]=H[l][k]=s;upper.push(s);}}
 const scale=H.map((row,k)=>Math.sqrt(Math.max(1e-10,row[k]))),A=H.map((row,k)=>row.map((v,l)=>v/(scale[k]*scale[l])+(k===l?audit.lambda:0)));
 let step=solve(A,grad.map((v,k)=>-v/scale[k])).map((v,k)=>v/scale[k]);const caps=[1,1,.2,.2,.25,20],ratio=Math.max(1,...step.map((v,k)=>Math.abs(v)/caps[k]));step=step.map(v=>v/ratio);
 const gpu=audit.blocks[j];checks.push({gaussian:j,gradient_relative_l2:rel(grad,gpu.g),hessian_relative_l2:rel(upper,gpu.H),step_max_abs:Math.max(...step.map((v,k)=>Math.abs(v-gpu.step[k])))});
}
const result={epsilon:1e-4,reference:'Float64 full-image central differences, no ROI truncation; normalized 6x6 Cholesky solve and identical step caps',blocks:checks,
 max_gradient_relative_l2:Math.max(...checks.map(x=>x.gradient_relative_l2)),max_hessian_relative_l2:Math.max(...checks.map(x=>x.hessian_relative_l2)),max_step_abs:Math.max(...checks.map(x=>x.step_max_abs)),
 tolerance:{gradient_relative_l2:1e-4,hessian_relative_l2:1e-4,step_max_abs:1e-3}};
result.passed=result.max_gradient_relative_l2<1e-4&&result.max_hessian_relative_l2<1e-4&&result.max_step_abs<1e-3;
fs.writeFileSync(new URL('results/desk_lm/linearization_check.json',root),JSON.stringify(result,null,2)+'\n');console.log(JSON.stringify({...result,blocks:undefined},null,2));assert.ok(result.passed);

import fs from 'node:fs';
import assert from 'node:assert/strict';
const root=new URL('../',import.meta.url),folder=new URL('results/desk_muon_pilot/',root);
// Independent FP64, transposed K x N formulation matching the primary source's NS5.
function ns5(g,N,K){
 let X=new Float64Array(K*N);let norm=0;
 for(let j=0;j<N;j++)for(let k=0;k<K;k++){const v=g[j*K+k];X[k*N+j]=v;norm+=v*v;}
 const den=Math.sqrt(norm)+1e-7;X=X.map(v=>v/den);
 for(let it=0;it<5;it++){
  const A=new Float64Array(K*K),B=new Float64Array(K*K),Y=new Float64Array(K*N);
  for(let k=0;k<K;k++)for(let l=0;l<K;l++)for(let j=0;j<N;j++)A[k*K+l]+=X[k*N+j]*X[l*N+j];
  for(let k=0;k<K;k++)for(let l=0;l<K;l++){
   let aa=0;for(let m=0;m<K;m++)aa+=A[k*K+m]*A[m*K+l];B[k*K+l]=-4.775*A[k*K+l]+2.0315*aa;
  }
  for(let k=0;k<K;k++)for(let j=0;j<N;j++){
   let v=0;for(let l=0;l<K;l++)v+=B[k*K+l]*X[l*N+j];Y[k*N+j]=3.4445*X[k*N+j]+v;
  }X=Y;
 }
 const out=new Float64Array(N*K),scale=Math.sqrt(Math.max(1,N/K));
 for(let j=0;j<N;j++)for(let k=0;k<K;k++)out[j*K+k]=X[k*N+j]*scale;
 return out;
}
const checks=[];
for(const file of fs.readdirSync(folder).filter(p=>p.endsWith('_audit.json'))){
 const r=JSON.parse(fs.readFileSync(new URL(file,folder),'utf8'));let max=0,diff2=0,ref2=0,momentumError=0;
 const K=r.mode===1?6:2,groups=r.mode===1?1:2;
 for(let c=0;c<r.chains;c++)for(let group=0;group<groups;group++){
  const g=new Float64Array(r.N*K);
  for(let j=0;j<r.N;j++)for(let k=0;k<K;k++){
   const at=(c*r.N+j)*6+group*K+k;
   g[j*K+k]=.05*r.scaled_gradient[at]+.95*r.momentum[at];
   if(r.steps===1)momentumError=Math.max(momentumError,Math.abs(r.momentum[at]-.05*r.scaled_gradient[at])/(1+Math.abs(r.momentum[at])));
  }
  const expected=ns5(g,r.N,K);
  for(let j=0;j<r.N;j++)for(let k=0;k<K;k++){
   const a=expected[j*K+k],b=r.update[(c*r.N+j)*6+group*K+k],d=Math.abs(a-b);max=Math.max(max,d);diff2+=d*d;ref2+=a*a;
  }
 }
 const relativeL2=Math.sqrt(diff2/ref2);assert.ok(relativeL2<2e-4);assert.ok(max<1e-3);assert.ok(momentumError<1e-6);
 checks.push({file,N:r.N,chains:r.chains,mode:r.mode,steps:r.steps,max_abs_error:max,relative_l2:relativeL2,first_step_momentum_error:momentumError,passed:true});
}
assert.equal(checks.length,4);
fs.writeFileSync(new URL('update_check.json',folder),JSON.stringify({passed:true,reference:'Independent CPU FP64 canonical transposed NS5; compare preconditioned direction before schedule and parameter projection.',tolerances:{relative_l2:2e-4,max_abs:1e-3},checks},null,2)+'\n');console.log(checks);

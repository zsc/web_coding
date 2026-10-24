export function policyFunction(flat,force=40){
 const h1=new Float64Array(16),h2=new Float64Array(16);
 return s=>{for(let j=0;j<16;j++){let z=flat[96+j];for(let k=0;k<6;k++)z+=flat[j*6+k]*s[k];h1[j]=Math.max(0,z);}for(let j=0;j<16;j++){let z=flat[368+j];for(let k=0;k<16;k++)z+=flat[112+j*16+k]*h1[k];h2[j]=Math.max(0,z);}let mu=flat[416];for(let k=0;k<16;k++)mu+=flat[384+k]*h2[k];return Math.max(-1,Math.min(1,mu))*force;};
}
export function assessPolicy(flat,force,model,{count=128,seed=2026,degrees=10,kicks=false}={}){
 let rng=seed>>>0;const random=()=>{rng=(Math.imul(1664525,rng)+1013904223)>>>0;return(rng+.5)/4294967296;};
 const control=policyFunction(flat,force);let success=0,survived=0,totalLength=0,totalReturn=0,maxForce=0;
 for(let ep=0;ep<count;ep++){
  let s=[(random()-.5)*.5,0,(random()*2-1)*degrees*model.RAD,0,(random()*2-1)*degrees*model.RAD,0],stable=0,failed=false,length=0,ret=0;
  const impulses=[random()<.5?-1:1,random()<.5?-1:1,random()<.5?-1:1];
  for(let t=0;t<500;t++){
   if(kicks&&(t===50||t===150||t===250))s[1]+=.35*impulses[(t-50)/100];
   const f=control(s);maxForce=Math.max(maxForce,Math.abs(f));s=model.step(s,f,force);length++;
   if(model.fallen(s)){failed=true;ret-=100;break;}
   ret+=1-4*(s[2]*s[2]+s[4]*s[4])-.1*s[0]*s[0]-.01*(s[1]*s[1]+s[3]*s[3]+s[5]*s[5]);
   stable=model.balanced(s)?stable+1:0;
  }success+=!failed&&stable>=50;survived+=!failed;totalLength+=length;totalReturn+=ret;
 }
 return{count,seed,degrees,kicks,success,successRate:success/count,survived,survivalRate:survived/count,meanLength:totalLength/count,meanReturn:totalReturn/count,maxForce};
}
export function assessSuite(flat,force,model,{count=512,seed=20261025}={}){
 return{engine:'Independent JavaScript Float64 physics and deterministic mean action',criterion:'10 s, no |angle|>60 deg or |x|>2.4m; final continuous 1 s: |angles|<2deg, |angular speeds|<0.15 rad/s, |cart speed|<0.15m/s',force,angle2:assessPolicy(flat,force,model,{count,seed,degrees:2}),angle5:assessPolicy(flat,force,model,{count,seed,degrees:5}),angle10:assessPolicy(flat,force,model,{count,seed,degrees:10}),pushes:assessPolicy(flat,force,model,{count,seed,degrees:2,kicks:true})};
}

/* Browser + Node shared runtime. SI units, absolute angles from upright.
 * Physical model matches ppo.py physical=True. No artificial stabilization.
 */
(function(root,factory){const api=factory();if(typeof module==='object'&&module.exports)module.exports=api;else root.PendulumModel=api;})(globalThis,()=>{
  'use strict';
  const DT=.02, RAD=Math.PI/180;
  const clamp=(x,a,b)=>Math.max(a,Math.min(b,x));
  function step(s,force,limit=10){
    const [x,v,t1,w1,t2,w2]=s,c1=Math.cos(t1),c2=Math.cos(t2),c12=Math.cos(t1-t2);
    const a=1.2,b=.15*c1,c=.05*c2,d=.1275,e=.05*c12,f=.0275;
    const n0=clamp(force,-limit,limit)+.15*Math.sin(t1)*w1*w1+.05*Math.sin(t2)*w2*w2;
    const n1=1.4715*Math.sin(t1)-.05*Math.sin(t1-t2)*w2*w2;
    const n2=.4905*Math.sin(t2)+.05*Math.sin(t1-t2)*w1*w1;
    const l10=b/a,l20=c/a,d1=d-b*l10,l21=(e-c*l10)/d1,d2=f-c*l20-l21*l21*d1;
    const y1=n1-l10*n0,y2=n2-l20*n0-l21*y1,z2=y2/d2,z1=y1/d1-l21*z2,z0=n0/a-l10*z1-l20*z2;
    const nv=v+DT*z0,nw1=w1+DT*z1,nw2=w2+DT*z2;
    return [x+DT*nv,nv,(t1+DT*nw1+Math.PI)%(2*Math.PI)-Math.PI,nw1,(t2+DT*nw2+Math.PI)%(2*Math.PI)-Math.PI,nw2];
  }
  function policy(p,s){for(let i=0;i<=4;i+=2){s=p[i+'.weight'].map((r,j)=>r.reduce((sum,w,k)=>sum+w*s[k],p[i+'.bias'][j]));if(i<4)s=s.map(x=>Math.max(0,x));}return 10*clamp(s[0],-1,1);}
  function lqr(k,s,limit=10){return clamp(-k.reduce((a,v,i)=>a+v*s[i],0),-limit,limit);}
  function fallen(s){return s.some(x=>!Number.isFinite(x))||Math.abs(s[0])>2.4||Math.max(Math.abs(s[2]),Math.abs(s[4]))>60*RAD;}
  function balanced(s){return Math.max(Math.abs(s[2]),Math.abs(s[4]))<2*RAD&&Math.max(Math.abs(s[3]),Math.abs(s[5]))<.15&&Math.abs(s[1])<.15;}
  function evaluate(controller,initial,steps=500,limit=10){
    let s=initial.slice(),stable=0,first=null,maxAngle=0,maxX=0;
    for(let i=0;i<steps;i++){
      s=step(s,controller(s),limit);maxAngle=Math.max(maxAngle,Math.abs(s[2]),Math.abs(s[4]));maxX=Math.max(maxX,Math.abs(s[0]));
      if(fallen(s))return {success:false,fell:true,steps:i+1,state:s,maxAngle,maxX};
      stable=balanced(s)?stable+1:0;
      if(stable===50&&first===null)first=(i+1)*DT;
    }
    return {success:stable>=50,fell:false,steps,state:s,recoverySeconds:first,maxAngle,maxX};
  }
  return {DT,RAD,clamp,step,policy,lqr,fallen,balanced,evaluate};
});

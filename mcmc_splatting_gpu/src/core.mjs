// Same image objective and clamped proposals as the upstream HTML.
export const keys = ['x','y','scaleX','scaleY','rotation','opacity'];
export function rng(seed=42) { let s=seed>>>0; return ()=>{s^=s<<13;s^=s>>>17;s^=s<<5;return (s>>>0)/4294967296;}; }
export function initialize(W,H,N,random) {
  return Array.from({length:N},()=>({x:random()*W,y:random()*H,scaleX:random()*W/10+W/20,scaleY:random()*H/10+H/20,rotation:random()*2*Math.PI,opacity:random()*100+50}));
}
export function propose(gs,W,H,random,steps=[2,2,1,1,.1,10]) {
  const index=Math.floor(random()*gs.length), param=Math.floor(random()*6);
  const g={...gs[index]};g[keys[param]]+=(random()-.5)*2*steps[param];
  g.x=Math.max(0,Math.min(W,g.x));g.y=Math.max(0,Math.min(H,g.y));
  g.scaleX=Math.max(.1,g.scaleX);g.scaleY=Math.max(.1,g.scaleY);g.opacity=Math.max(0,Math.min(255,g.opacity));
  return {index,g,u:random()};
}
export function accept(delta,T,u) { return u<Math.min(1,Math.exp(-delta/T)); }
export const quantize = s => Math.min(255,Math.max(0,Math.round(s)));
export function contribution(g,W,H,out) {
  const c=Math.cos(-g.rotation),s=Math.sin(-g.rotation),sx=Math.max(.1,g.scaleX),sy=Math.max(.1,g.scaleY);
  for(let y=0,p=0;y<H;y++)for(let x=0;x<W;x++,p++){
    const dx=x-g.x,dy=y-g.y,a=(dx*c-dy*s)/sx,b=(dx*s+dy*c)/sy;
    out[p]=g.opacity*Math.exp(-.5*(a*a+b*b));
  }
}
export class CachedChain {
  constructor(gs,W,H,target) {
    this.gs=structuredClone(gs);this.W=W;this.H=H;this.target=target;this.P=W*H;
    this.cache=gs.map(()=>new Float64Array(this.P));this.sum=new Float64Array(this.P);
    this.candidate=new Float64Array(this.P);this.accepted=0;this.iter=0;this.rebuild();
  }
  rebuild() {
    this.sum.fill(0);
    this.gs.forEach((g,j)=>{contribution(g,this.W,this.H,this.cache[j]);for(let p=0;p<this.P;p++)this.sum[p]+=this.cache[j][p];});
    this.sse=0;for(let p=0;p<this.P;p++)this.sse+=(quantize(this.sum[p])-this.target[p])**2;
  }
  step(proposal,T) {
    const {index,g,u}=proposal,old=this.cache[index];
    contribution(g,this.W,this.H,this.candidate);
    let sse=0;for(let p=0;p<this.P;p++)sse+=(quantize(this.sum[p]-old[p]+this.candidate[p])-this.target[p])**2;
    // Difference of MSEs follows upstream arithmetic, including the T=0/NaN case.
    const yes=accept(sse/this.P-this.sse/this.P,T,u);
    if(yes){for(let p=0;p<this.P;p++)this.sum[p]=this.sum[p]-old[p]+this.candidate[p];
      this.cache[index]=this.candidate;this.candidate=old;this.gs[index]=g;this.sse=sse;this.accepted++;}
    this.iter++; return yes;
  }
  get loss(){return this.sse/this.P;}
  image(){return Uint8Array.from(this.sum,quantize);}
}
// Portable, deterministic test image; no OS font or browser antialiasing dependency.
export function targetFixture(W,H) {
  const glyphs=['10001/11011/10101/10101/10001/10001/10001','01111/10000/10000/10000/10000/10000/01111','10001/11011/10101/10101/10001/10001/10001','01111/10000/10000/10000/10000/10000/01111'].map(x=>x.split('/'));
  const a=new Uint8Array(W*H),scale=Math.min(W/27,H/11),ox=(W-23*scale)/2,oy=(H-7*scale)/2;
  for(let y=0;y<H;y++)for(let x=0;x<W;x++){
    let v=0;for(let sy=0;sy<2;sy++)for(let sx=0;sx<2;sx++){
      const gx=Math.floor((x+(sx+.5)/2-ox)/scale),gy=Math.floor((y+(sy+.5)/2-oy)/scale),i=Math.floor(gx/6),col=gx%6;
      if(i>=0&&i<4&&col<5&&col>=0&&gy>=0&&gy<7&&glyphs[i][gy][col]==='1')v++;
    }a[y*W+x]=Math.round(v*255/4);
  }return a;
}

import fs from 'node:fs';
import {accept} from './core.mjs';
const html=fs.readFileSync(new URL('../source/mcmc_splatting.original.html',import.meta.url),'utf8');
// Execute upstream renderer and loss function verbatim in the same JS realm.
function extract(name) {
  const start=html.indexOf('function '+name+'('),open=html.indexOf('{',start);let depth=1,i=open+1;
  for(;depth;i++){if(html[i]==='{')depth++;else if(html[i]==='}')depth--;}
  return html.slice(start,i);
}
const factory=new Function('splatCtx',extract('renderGaussiansToImageData')+'\n'+extract('calculateLoss')+'\nreturn {renderGaussiansToImageData,calculateLoss};');
export const original=factory({createImageData:(W,H)=>({data:new Uint8ClampedArray(W*H*4)})});
export class ReferenceChain {
  constructor(gs,W,H,target){this.gs=structuredClone(gs);this.W=W;this.H=H;this.accepted=0;this.iter=0;
    this.target={data:new Uint8ClampedArray(W*H*4)};target.forEach((v,i)=>this.target.data[i*4]=v);
    this.rendered=original.renderGaussiansToImageData(this.gs,W,H);this.loss=original.calculateLoss(this.rendered,this.target);}
  step({index,g,u},T){const candidate=JSON.parse(JSON.stringify(this.gs));candidate[index]=g;
    const image=original.renderGaussiansToImageData(candidate,this.W,this.H),loss=original.calculateLoss(image,this.target);
    const yes=accept(loss-this.loss,T,u);if(yes){this.gs=candidate;this.rendered=image;this.loss=loss;this.accepted++;}this.iter++;return yes;}
  image(){return Uint8Array.from({length:this.W*this.H},(_,p)=>this.rendered.data[p*4]);}
}

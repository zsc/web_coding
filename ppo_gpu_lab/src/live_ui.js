/* Offline UI. Only forces from the selected controller enter the physics step. */
(()=>{
  'use strict';
  const M=PendulumModel,data=JSON.parse(document.getElementById('live-data').textContent);
  const el=id=>document.getElementById(id),canvas=el('liveCanvas'),ctx=canvas.getContext('2d');
  let state=[0,0,3*M.RAD,0,3*M.RAD,0],time=0,force=0,auto=true,mode='lqr',limit=40;
  let running=!matchMedia('(prefers-reduced-motion: reduce)').matches,failed=false,drag=null;
  let stableSteps=0,recoveryStart=0,recoveredAt=null,accumulator=0,last=null,trace=[];
  const PX=130,BASE=400,CENTER=500;
  const initial=()=>[0,0,0,0,0,0];
  function geometry(){const x=CENTER+state[0]*PX,p0={x,y:BASE},p1={x:x+Math.sin(state[2])*PX,y:BASE-Math.cos(state[2])*PX};return[p0,p1,{x:p1.x+Math.sin(state[4])*PX,y:p1.y-Math.cos(state[4])*PX}];}
  function restartRecovery(){stableSteps=0;recoveryStart=time;recoveredAt=null;}
  function setState(s){state=s;time=0;force=0;failed=false;trace=[];accumulator=0;restartRecovery();render();}
  function applyPreset(){if(drag)return;const a=+el('liveAngle').value*M.RAD;setState([0,0,a,0,a,0]);}
  function control(){return !auto?0:mode==='lqr'?M.lqr(data.lqr.K,state,limit):M.policy(data.policy,state);}
  function tick(){
    force=control();state=M.step(state,force,limit);time+=M.DT;
    if(M.fallen(state)){failed=true;force=0;return;}
    stableSteps=M.balanced(state)?stableSteps+1:0;
    if(stableSteps===50&&recoveredAt===null)recoveredAt=time-recoveryStart;
    const g=geometry();trace.push(g[2]);if(trace.length>130)trace.shift();
  }
  function push(direction){if(failed||drag)return;state[1]+=direction*.35;restartRecovery();render();}
  function pause(){running=!running;accumulator=0;last=null;render();}
  function status(){
    if(drag)return ['正在拖动','drag'];
    if(failed)return ['未接住 · 请重置','failed'];
    if(!running)return ['已暂停','paused'];
    if(!auto)return ['自由运动 · 控制已关','off'];
    if(stableSteps>=50)return ['已稳定','balanced'];
    return ['正在恢复','recovering'];
  }
  function line(a,b,color,width){ctx.beginPath();ctx.moveTo(a.x,a.y);ctx.lineTo(b.x,b.y);ctx.strokeStyle=color;ctx.lineWidth=width;ctx.stroke();}
  function circle(p,r,color){ctx.fillStyle=color;ctx.beginPath();ctx.arc(p.x,p.y,r,0,Math.PI*2);ctx.fill();}
  function render(){
    const [kind,label]=[status()[1],status()[0]];
    if(el('liveStatus').textContent!==label)el('liveStatus').textContent=label;
    el('liveStatus').dataset.kind=kind;
    el('liveModeTag').textContent=`${auto?(mode==='lqr'?'LQR':'PPO'):'无控制'} · ±${limit} N`;
    el('livePause').textContent=running?'暂停':'继续';el('liveAuto').textContent=auto?'自动 · 开':'自动 · 关';el('liveAuto').setAttribute('aria-pressed',String(auto));
    el('livePower').disabled=mode==='ppo';
    el('liveControllerNote').textContent=mode==='ppo'?'使用 GPU 已训练权重，力上限固定 10 N。只训练过竖直附近的小扰动，较大拖拽可能失败。':limit===40?'根据当前姿态实时计算推力。40 N 是交互增强设置，不属于原 PPO 基准。':'原任务的 10 N 力上限。较大倾斜可能因电机饱和而无法恢复。';
    el('liveHint').textContent=drag?'松手后从静止释放 · 按住期间暂停物理':failed?'这次超出了恢复范围，点「重置」再试一次':!running?'已暂停 · 可以调整姿态，点击「继续」运行':'拖动圆形关节或小车 · 空格暂停 / R 重置';
    el('liveClock').textContent=time.toFixed(2)+' s';
    el('liveTheta1').textContent=(state[2]/M.RAD).toFixed(1)+'°';el('liveTheta2').textContent=(state[4]/M.RAD).toFixed(1)+'°';el('liveX').textContent=state[0].toFixed(2)+' m';el('liveForce').textContent=force.toFixed(1)+' N';
    el('liveRecovery').textContent=failed?'没有自动重置。保留失败姿态供观察。':recoveredAt!==null&&stableSteps>=50?`释放后 ${recoveredAt.toFixed(2)} s 达到连续 1 s 稳定。`:drag?'松开后开始计算恢复时间。':'保持稳定 1 秒后，标记为已稳定。';
    const bar=el('liveForceBar'),f=force/limit;bar.style.left=(f<0?50+f*50:50)+'%';bar.style.width=Math.abs(f)*50+'%';
    el('pushLeft').disabled=el('pushRight').disabled=failed;
    // DOM diagnostics also describe the exact displayed state for reproducible UI QA.
    canvas.dataset.state=JSON.stringify(state);canvas.dataset.time=time.toFixed(2);canvas.dataset.status=kind;
    ctx.clearRect(0,0,1000,560);ctx.lineCap='round';
    ctx.strokeStyle='#264339';ctx.lineWidth=1;
    for(let x=45;x<1000;x+=65)line({x,y:45},{x,y:470},'#264339',1);
    for(let y=75;y<490;y+=65)line({x:32,y},{x:968,y},'#264339',1);
    const [p0,p1,p2]=geometry();
    ctx.setLineDash([3,8]);line({x:p0.x,y:76},p0,'#466351',1);ctx.setLineDash([]);
    line({x:CENTER-2.4*PX,y:BASE+38},{x:CENTER+2.4*PX,y:BASE+38},'#6f8a72',3);
    for(const x of[-2.4,0,2.4]){const xx=CENTER+x*PX;line({x:xx,y:BASE+33},{x:xx,y:BASE+43},'#789980',2);ctx.fillStyle='#8eab98';ctx.font='13px ui-monospace,monospace';ctx.textAlign='center';ctx.fillText(x===0?'中心':x.toFixed(1)+' m',xx,BASE+66);}
    if(trace.length>1){ctx.strokeStyle='#9bd4ac33';ctx.lineWidth=2;ctx.beginPath();trace.forEach((p,i)=>i?ctx.lineTo(p.x,p.y):ctx.moveTo(p.x,p.y));ctx.stroke();}
    line(p0,p1,'#8dcd9a',12);line(p1,p2,'#eac38b',10);
    ctx.fillStyle='#75a98a';ctx.beginPath();ctx.roundRect(p0.x-42,BASE-12,84,36,8);ctx.fill();
    for(const x of[p0.x-25,p0.x+25])circle({x,y:BASE+30},8,'#c6ddc6');
    circle(p0,7,'#d9edd1');
    for(const [i,p]of[[1,p1],[2,p2]]){circle(p,24,drag?.part===i?'#ffe2aa25':'#d5e9d510');ctx.lineWidth=1;ctx.strokeStyle=drag?.part===i?'#ffe2aa':'#8baf9380';ctx.beginPath();ctx.arc(p.x,p.y,24,0,2*Math.PI);ctx.stroke();circle(p,10,i===1?'#b6dfb7':'#f1d2a0');circle(p,3,'#1c3b2c');}
    ctx.textAlign='left';ctx.font='13px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif';ctx.fillStyle='#9abca4';ctx.fillText('下杆 θ₁',p1.x+30,p1.y+4);ctx.fillStyle='#c9b38f';ctx.fillText('上杆 θ₂',p2.x+30,p2.y+4);
    if(Math.abs(force)>.15){const start={x:p0.x,y:BASE+91},end={x:p0.x+force/limit*100,y:BASE+91};line(start,end,'#a4d9b4',4);line(end,{x:end.x-Math.sign(force)*10,y:end.y-6},'#a4d9b4',3);line(end,{x:end.x-Math.sign(force)*10,y:end.y+6},'#a4d9b4',3);}
    if(failed){ctx.fillStyle='#172e26d9';ctx.beginPath();ctx.roundRect(280,30,440,54,8);ctx.fill();ctx.textAlign='center';ctx.fillStyle='#edc197';ctx.font='18px -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif';ctx.fillText('超出恢复范围 · 重置后再试',500,64);}
  }
  function point(e){const r=canvas.getBoundingClientRect();return{x:(e.clientX-r.left)*1000/r.width,y:(e.clientY-r.top)*560/r.height};}
  function distance(p,a,b){const dx=b.x-a.x,dy=b.y-a.y,t=M.clamp(((p.x-a.x)*dx+(p.y-a.y)*dy)/(dx*dx+dy*dy),0,1);return Math.hypot(p.x-a.x-dx*t,p.y-a.y-dy*t);}
  canvas.addEventListener('pointerdown',e=>{
    if(drag||failed||e.button!==0)return;
    const p=point(e),g=geometry(),radius=Math.max(26,18*1000/canvas.getBoundingClientRect().width);
    let part=null;
    if(Math.hypot(p.x-g[2].x,p.y-g[2].y)<radius)part=2;
    else if(Math.hypot(p.x-g[1].x,p.y-g[1].y)<radius)part=1;
    else if(Math.abs(p.x-g[0].x)<60&&Math.abs(p.y-BASE)<45)part=0;
    else if(distance(p,g[1],g[2])<radius*.65)part=2;
    else if(distance(p,g[0],g[1])<radius*.65)part=1;
    if(part===null)return;
    e.preventDefault();canvas.focus({preventScroll:true});canvas.setPointerCapture(e.pointerId);
    drag={part,id:e.pointerId,offset:p.x-g[0].x};canvas.classList.add('dragging');force=0;
    state[1]=state[3]=state[5]=0;trace=[];restartRecovery();accumulator=0;render();
  });
  canvas.addEventListener('pointermove',e=>{
    if(!drag||drag.id!==e.pointerId)return;e.preventDefault();const p=point(e),g=geometry();
    if(drag.part===0)state[0]=M.clamp((p.x-drag.offset-CENTER)/PX,-2.2,2.2);
    else{const pivot=g[drag.part-1];state[drag.part===1?2:4]=M.clamp(Math.atan2(p.x-pivot.x,pivot.y-p.y),-55*M.RAD,55*M.RAD);}
    render();
  });
  function release(e){if(!drag||drag.id!==e.pointerId)return;drag=null;canvas.classList.remove('dragging');if(canvas.hasPointerCapture(e.pointerId))canvas.releasePointerCapture(e.pointerId);state[1]=state[3]=state[5]=0;accumulator=0;last=null;restartRecovery();render();}
  canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);canvas.addEventListener('lostpointercapture',release);
  canvas.addEventListener('keydown',e=>{if(e.code==='Space'){e.preventDefault();pause();}else if(e.key.toLowerCase()==='r')setState(initial());else if(e.key==='ArrowLeft'){e.preventDefault();push(-1);}else if(e.key==='ArrowRight'){e.preventDefault();push(1);}});
  el('livePause').addEventListener('click',pause);el('liveReset').addEventListener('click',()=>setState(initial()));
  el('pushLeft').addEventListener('click',()=>push(-1));el('pushRight').addEventListener('click',()=>push(1));
  el('liveAuto').addEventListener('click',()=>{auto=!auto;force=0;restartRecovery();render();});
  el('liveController').addEventListener('change',e=>{mode=e.target.value;if(mode==='ppo'){limit=10;el('livePower').value='10';}force=0;restartRecovery();render();});
  el('livePower').addEventListener('change',e=>{limit=+e.target.value;force=0;restartRecovery();render();});
  el('liveAngle').addEventListener('input',e=>el('liveAngleValue').textContent=e.target.value+'°');el('liveApply').addEventListener('click',applyPreset);
  document.addEventListener('visibilitychange',()=>{last=null;accumulator=0;});
  const names=['ppo10','lqr10','lqr40'];
  el('liveEvidenceRows').innerHTML=data.evaluation.grid.map(row=>`<tr><td>±${row.degrees}°</td>${names.map(k=>`<td>${row[k].success} / 8</td>`).join('')}</tr>`).join('');
  const random=data.evaluation.random.lqr40;
  el('liveEvidenceNote').textContent=`另外使用独立 LCG seed 20261024 采样 512 组初始状态：两角各在 ±10° 内，位置 ±0.25 m，速度为零。LQR 40 N 成功 ${random.success} / ${random.total}。这些是该模拟器的实测结果。`;
  function frame(ts){if(last===null)last=ts;const elapsed=Math.min(.1,(ts-last)/1000);last=ts;
    if(running&&!failed&&!drag&&!document.hidden){accumulator+=elapsed;while(accumulator+1e-10>=M.DT){tick();accumulator-=M.DT;if(failed)break;}}else accumulator=0;
    render();requestAnimationFrame(frame);
  }
  render();requestAnimationFrame(frame);
})();

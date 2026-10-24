// Block-damped Gauss-Newton with whole-chain trial acceptance.
// Each convex quadratic neglects cross-Gaussian Hessian blocks. Global trial
// rendering handles that approximation; this is not an exact global LM solve.
#define ROI_LIBRARY_ONLY
#include "roi_fit.cu"
#include <limits>

struct LMState { double loss; float damping; int accepted,rejected,take; };
struct BlockAudit { float gradient[6],hessian[21],delta[6]; };

__global__ void continuousLoss(const float* sum,const unsigned char* target,double* loss,int P){
 int c=blockIdx.x,t=threadIdx.x;double s=0;
 for(int p=t;p<P;p+=THREADS){double r=double(sum[c*P+p])-target[p];s+=r*r;}
 for(int d=16;d;d/=2)s+=__shfl_down_sync(0xffffffff,s,d);
 __shared__ double part[4];if(t%32==0)part[t/32]=s;__syncthreads();
 if(t==0)loss[c]=(part[0]+part[1]+part[2]+part[3])/P;
}
__global__ void startLM(LMState* state,const double* loss,double* history,int C,float damping){
 int c=threadIdx.x+blockIdx.x*blockDim.x;if(c<C){state[c]={loss[c],damping,0,0,0};history[c]=loss[c];}
}

__global__ void blockProposal(const Packed* current,Packed* candidate,const float* sum,
 const unsigned char* target,const LMState* states,int W,int H,int N,BlockAudit* audit){
 int j=blockIdx.x,c=blockIdx.y,t=threadIdx.x,index=c*N+j;Packed g=current[index];
 int x0=max(0,int(floorf(g.x-g.rx))),y0=max(0,int(floorf(g.y-g.ry)));
 int x1=min(W-1,int(ceilf(g.x+g.rx))),y1=min(H-1,int(ceilf(g.y+g.ry)));
 int rw=max(0,x1-x0+1),rh=max(0,y1-y0+1);
 float stats[27]={0};
 for(int q=t;q<rw*rh;q+=THREADS){
  int x=x0+q%rw,y=y0+q/rw,p=y*W+x;float dx=x-g.x,dy=y-g.y;
  float u=dx*g.co-dy*g.si,v=dx*g.si+dy*g.co,ux=u*g.ix,vy=v*g.iy,qq=ux*ux+vy*vy;
  if(qq<=36){
   float e=expf(-.5f*qq),f=g.a*e,r=sum[c*W*H+p]-target[p],ix2=g.ix*g.ix,iy2=g.iy*g.iy;
   float J[6]={f*(u*g.co*ix2+v*g.si*iy2),f*(-u*g.si*ix2+v*g.co*iy2),f*ux*ux,f*vy*vy,f*u*v*(iy2-ix2),e};
   #pragma unroll
   for(int k=0;k<6;k++)stats[k]+=J[k]*r;
   int at=6;
   #pragma unroll
   for(int k=0;k<6;k++){
    #pragma unroll
    for(int l=k;l<6;l++)stats[at++]+=J[k]*J[l];
   }
  }
 }
 __shared__ float partial[108];
 #pragma unroll
 for(int k=0;k<27;k++){float s=stats[k];for(int d=16;d;d/=2)s+=__shfl_down_sync(0xffffffff,s,d);if(t%32==0)partial[(t/32)*27+k]=s;}
 __syncthreads();
 if(t==0){
  for(int k=0;k<27;k++)stats[k]=partial[k]+partial[27+k]+partial[54+k]+partial[81+k];
  float A[36]={0},scale[6],rhs[6],step[6]={0};int at=6;
  for(int k=0;k<6;k++)for(int l=k;l<6;l++){A[k*6+l]=A[l*6+k]=stats[at++];}
  // Diagonal equilibration avoids mixing pixels, log-scales, radians and intensity.
  // The original regularizer is lambda * diag(max(H_kk,1e-10)).
  for(int k=0;k<6;k++)scale[k]=sqrtf(fmaxf(A[k*6+k],1e-10f));
  for(int k=0;k<6;k++){rhs[k]=-stats[k]/scale[k];for(int l=0;l<6;l++)A[k*6+l]/=scale[k]*scale[l];A[k*6+k]+=states[c].damping;}
  bool ok=true;
  for(int k=0;k<6;k++){
   for(int l=0;l<=k;l++){float s=A[k*6+l];for(int m=0;m<l;m++)s-=A[k*6+m]*A[l*6+m];
    if(k==l){if(!(s>0)||!isfinite(s)){ok=false;s=1;}A[k*6+l]=sqrtf(s);}
    else A[k*6+l]=s/A[l*6+l];
   }
  }
  float z[6]={0};
  for(int k=0;k<6;k++){float v=rhs[k];for(int l=0;l<k;l++)v-=A[k*6+l]*z[l];z[k]=v/A[k*6+k];}
  for(int k=5;k>=0;k--){float v=z[k];for(int l=k+1;l<6;l++)v-=A[l*6+k]*step[l];step[k]=v/A[k*6+k];}
  for(int k=0;k<6;k++){step[k]/=scale[k];if(!isfinite(step[k]))ok=false;}
  // Bound each Gaussian's step uniformly before applying box constraints.
  const float caps[6]={1.f,1.f,.2f,.2f,.25f,20.f};float ratio=1;
  for(int k=0;k<6;k++)ratio=fmaxf(ratio,fabsf(step[k])/caps[k]);
  for(int k=0;k<6;k++)step[k]=ok?step[k]/ratio:0.f;
  if(audit){for(int k=0;k<6;k++){audit[index].gradient[k]=stats[k];audit[index].delta[k]=step[k];}for(int k=0;k<21;k++)audit[index].hessian[k]=stats[k+6];}
  Gaussian<float> out{{bound(g.x+step[0],0.f,float(W)),bound(g.y+step[1],0.f,float(H)),
    bound(g.sx*expf(step[2]),.1f,32.f),bound(g.sy*expf(step[3]),.1f,32.f),g.angle+step[4],bound(g.a+step[5],0.f,255.f)}};
  candidate[index]=packed(out);
 }
}

__global__ void decideLM(LMState* states,const double* candidateLoss,double* history,int C,int iteration){
 int c=blockIdx.x*blockDim.x+threadIdx.x;if(c>=C)return;
 auto s=states[c];double next=candidateLoss[c];s.take=isfinite(next)&&next<s.loss;
 if(s.take){s.loss=next;s.accepted++;s.damping=fmaxf(1e-4f,s.damping*.7f);}
 else{s.rejected++;s.damping=fminf(1e8f,s.damping*4.f);}
 states[c]=s;history[(iteration+1)*C+c]=s.loss;
}
__global__ void commitLM(Packed* current,const Packed* candidate,float* sum,const float* candidateSum,const LMState* states,int N,int P){
 int p=blockIdx.x*blockDim.x+threadIdx.x,c=blockIdx.y;if(!states[c].take)return;
 if(p<P)sum[c*P+p]=candidateSum[c*P+p];if(p<N)current[c*N+p]=candidate[c*N+p];
}

int main(int argc,char** argv){try{
 auto entire=Clock::now();std::string input="zhou_target.pgm",out="results/lm",auditPath;
 int N=1600,C=8,adamSteps=0,lmSteps=1000,seed=42;float damping=1;
 for(int i=1;i<argc;i++){std::string a=argv[i];if(i+1>=argc)throw std::runtime_error("missing value");std::string v=argv[++i];
  if(a=="--input")input=v;else if(a=="--output")out=v;else if(a=="--gaussians")N=std::stoi(v);else if(a=="--chains")C=std::stoi(v);
  else if(a=="--adam-steps")adamSteps=std::stoi(v);else if(a=="--lm-steps")lmSteps=std::stoi(v);else if(a=="--lambda")damping=std::stof(v);
  else if(a=="--seed")seed=std::stoi(v);else if(a=="--audit")auditPath=v;else throw std::runtime_error("unknown option: "+a);
 }
 if(N<1||N>4096||C<1||C>128||adamSteps<0||lmSteps<0||adamSteps>1000000||lmSteps>1000000||!std::isfinite(damping)||damping<=0||(!adamSteps&&!lmSteps))throw std::runtime_error("invalid configuration");
 if(!auditPath.empty()&&adamSteps)throw std::runtime_error("audit requires zero Adam steps");
 int W=0,H=0;auto target=readPgm(input,W,H);int P=W*H;if(P>65536)throw std::runtime_error("image too large");
 std::vector<double> cdf(P);double mass=0;for(int p=0;p<P;p++)cdf[p]=(mass+=target[p]);if(mass==0)throw std::runtime_error("empty target");
 std::vector<Packed> gs(C*N);
 for(int c=0;c<C;c++){uint32_t r=uint32_t(seed+c*1009);for(int j=0;j<N;j++){
  int p=std::lower_bound(cdf.begin(),cdf.end(),(j+uniform(r))*mass/N)-cdf.begin();float sx=1.1f+uniform(r)*1.5f,sy=1.1f+uniform(r)*1.5f,angle=uniform(r)*6.283185307179586;
  gs[c*N+j]=packed(Gaussian<float>{{float(p%W),float(p/W),sx,sy,angle,float(std::min(230.,mass/(N*6.283185307179586*sx*sy)))}});
 }}
 auto initial=gs;size_t bytes=gs.size()*(2*sizeof(Packed)+12*sizeof(float))+size_t(C)*P*8+P+C*(sizeof(LMState)+sizeof(double)+sizeof(long long))+size_t(C)*(lmSteps+1)*sizeof(double)+(auditPath.empty()?0:gs.size()*sizeof(BlockAudit));
 size_t free,total;CK(cudaMemGetInfo(&free,&total));if(bytes>free/2)throw std::runtime_error("VRAM budget exceeded");
 auto dg=alloc<Packed>(gs.size()),trial=alloc<Packed>(gs.size());auto sum=alloc<float>(C*P),trialSum=alloc<float>(C*P),moments=alloc<float>(gs.size()*12);
 auto dt=alloc<unsigned char>(P);auto loss=alloc<double>(C),history=alloc<double>(size_t(C)*(lmSteps+1));auto states=alloc<LMState>(C);auto quantized=alloc<long long>(C);
 BlockAudit* audit=auditPath.empty()?nullptr:alloc<BlockAudit>(gs.size());
 CK(cudaMemcpy(dg,gs.data(),gs.size()*sizeof(Packed),cudaMemcpyHostToDevice));CK(cudaMemcpy(dt,target.data(),P,cudaMemcpyHostToDevice));CK(cudaMemset(moments,0,gs.size()*48));
 CK(cudaMemset(sum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(dg,sum,W,H,N);CK(cudaGetLastError());CK(cudaDeviceSynchronize());double setup=elapsed(entire);
 cudaEvent_t begin,adamEnd,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&adamEnd));CK(cudaEventCreate(&end));CK(cudaEventRecord(begin));
 for(int a=0;a<adamSteps;a++){
  CK(cudaMemsetAsync(sum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(dg,sum,W,H,N);
  refineAdam<<<dim3(N,C),THREADS>>>(dg,sum,dt,moments,moments+gs.size()*6,W,H,N,a,adamSteps);
 }
 CK(cudaMemsetAsync(sum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(dg,sum,W,H,N);
 continuousLoss<<<C,THREADS>>>(sum,dt,loss,P);startLM<<<(C+127)/128,128>>>(states,loss,history,C,damping);CK(cudaEventRecord(adamEnd));
 for(int a=0;a<lmSteps;a++){
  blockProposal<<<dim3(N,C),THREADS>>>(dg,trial,sum,dt,states,W,H,N,(a==0?audit:nullptr));
  CK(cudaMemsetAsync(trialSum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(trial,trialSum,W,H,N);
  continuousLoss<<<C,THREADS>>>(trialSum,dt,loss,P);decideLM<<<(C+127)/128,128>>>(states,loss,history,C,a);
  commitLM<<<dim3((std::max(N,P)+127)/128,C),128>>>(dg,trial,sum,trialSum,states,N,P);
 }
 sparseLoss<<<C,THREADS>>>(sum,quantized,dt,P);CK(cudaGetLastError());CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float gpuMs,adamMs;CK(cudaEventElapsedTime(&gpuMs,begin,end));CK(cudaEventElapsedTime(&adamMs,begin,adamEnd));
 std::vector<long long> scores(C);std::vector<LMState> finalState(C);std::vector<double> curve(size_t(C)*(lmSteps+1));
 CK(cudaMemcpy(gs.data(),dg,gs.size()*sizeof(Packed),cudaMemcpyDeviceToHost));CK(cudaMemcpy(scores.data(),quantized,C*8,cudaMemcpyDeviceToHost));
 CK(cudaMemcpy(finalState.data(),states,C*sizeof(LMState),cudaMemcpyDeviceToHost));CK(cudaMemcpy(curve.data(),history,curve.size()*8,cudaMemcpyDeviceToHost));
 int winner=std::min_element(scores.begin(),scores.end())-scores.begin();std::vector<float> cached(P);CK(cudaMemcpy(cached.data(),sum+winner*P,P*4,cudaMemcpyDeviceToHost));
 std::vector<double> full(P);for(int j=0;j<N;j++){auto g=gs[winner*N+j];Gaussian<double> ref{{g.x,g.y,g.sx,g.sy,g.angle,g.a}};double co=cos(-ref.v[4]),si=sin(-ref.v[4]);for(int p=0;p<P;p++)full[p]+=value(ref,p%W,p/W,co,si);}
 long long sse=0,cachedSSE=0;double smooth=0,pixelMae=0,maxSumDiff=0;std::vector<unsigned char> image(P);
 for(int p=0;p<P;p++){if(!std::isfinite(full[p]))throw std::runtime_error("nonfinite render");image[p]=quantize(full[p]);long long d=int(image[p])-target[p];sse+=d*d;d=quantize(cached[p])-target[p];cachedSSE+=d*d;smooth+=(full[p]-target[p])*(full[p]-target[p]);pixelMae+=abs(int(image[p])-quantize(cached[p]));maxSumDiff=std::max(maxSumDiff,abs(full[p]-cached[p]));}
 if(cachedSSE!=scores[winner])throw std::runtime_error("quantized loss mismatch");
 for(int a=1;a<=lmSteps;a++)for(int c=0;c<C;c++)if(!(curve[a*C+c]<=curve[(a-1)*C+c]))throw std::runtime_error("LM acceptance increased smooth loss");
 std::ofstream pgm(out+".pgm",std::ios::binary);if(!pgm)throw std::runtime_error("cannot write output");pgm<<"P5\n"<<W<<" "<<H<<"\n255\n";pgm.write((char*)image.data(),P);pgm.close();
 if(audit){std::vector<BlockAudit> rows(gs.size());CK(cudaMemcpy(rows.data(),audit,rows.size()*sizeof(BlockAudit),cudaMemcpyDeviceToHost));std::ofstream f(auditPath);f<<std::setprecision(12)<<"{\"lambda\":"<<damping<<",\"W\":"<<W<<",\"H\":"<<H<<",\"N\":"<<N<<",\"chains\":"<<C<<",\"initial\":[";
  for(size_t j=0;j<initial.size();j++){if(j)f<<",";auto g=initial[j];f<<"["<<g.x<<","<<g.y<<","<<g.sx<<","<<g.sy<<","<<g.angle<<","<<g.a<<"]";}f<<"],\"blocks\":[";
  for(size_t j=0;j<rows.size();j++){if(j)f<<",";auto r=rows[j];f<<"{\"g\":[";for(int k=0;k<6;k++){if(k)f<<",";f<<r.gradient[k];}f<<"],\"H\":[";for(int k=0;k<21;k++){if(k)f<<",";f<<r.hessian[k];}f<<"],\"step\":[";for(int k=0;k<6;k++){if(k)f<<",";f<<r.delta[k];}f<<"]}";}f<<"]}\n";
 }
 std::ofstream json(out+".json");if(!json)throw std::runtime_error("cannot write JSON");json<<std::setprecision(12)<<"{\"backend\":\"block_damped_gauss_newton\",\"W\":"<<W<<",\"H\":"<<H<<",\"N\":"<<N<<",\"chains\":"<<C<<",\"seed\":"<<seed<<",\"adam_steps\":"<<adamSteps<<",\"lm_steps\":"<<lmSteps<<",\"initial_lambda\":"<<damping<<",\"gpu_ms\":"<<gpuMs<<",\"adam_gpu_ms\":"<<adamMs<<",\"lm_gpu_ms\":"<<gpuMs-adamMs<<",\"setup_ms\":"<<setup<<",\"end_to_end_ms\":"<<elapsed(entire)<<",\"loss\":"<<double(sse)/P<<",\"smooth_loss\":"<<smooth/P<<",\"cached_loss\":"<<double(scores[winner])/P<<",\"pixel_mae\":"<<pixelMae/P<<",\"max_sum_drift\":"<<maxSumDiff<<",\"workspace_bytes\":"<<bytes<<",\"winner\":"<<winner<<",\"monotonic_smooth_loss\":true,\"lm_states\":[";
 for(int c=0;c<C;c++){if(c)json<<",";auto s=finalState[c];json<<"{\"loss\":"<<s.loss<<",\"lambda\":"<<s.damping<<",\"accepted\":"<<s.accepted<<",\"rejected\":"<<s.rejected<<"}";}json<<"],\"curve_stride\":10,\"smooth_curve\":[";
 for(int a=0;a<=lmSteps;a++){if(a%10&&a!=lmSteps)continue;if(a)json<<",";json<<"{\"step\":"<<a<<",\"losses\":[";for(int c=0;c<C;c++){if(c)json<<",";json<<curve[a*C+c];}json<<"]}";}json<<"],\"gaussians\":[";
 for(int j=0;j<N;j++){if(j)json<<",";auto g=gs[winner*N+j];json<<"["<<g.x<<","<<g.y<<","<<g.sx<<","<<g.sy<<","<<g.angle<<","<<g.a<<"]";}json<<"]}\n";json.close();
 std::cout<<"Adam "<<adamSteps<<" + block LM "<<lmSteps<<": GPU "<<gpuMs<<" ms, end "<<elapsed(entire)<<" ms, MSE "<<double(sse)/P<<", accepted "<<finalState[winner].accepted<<"/"<<lmSteps<<"\n";
 for(void* p:{(void*)dg,(void*)trial,(void*)sum,(void*)trialSum,(void*)moments,(void*)dt,(void*)loss,(void*)history,(void*)states,(void*)quantized,(void*)audit})if(p)CK(cudaFree(p));CK(cudaEventDestroy(begin));CK(cudaEventDestroy(adamEnd));CK(cudaEventDestroy(end));
 return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

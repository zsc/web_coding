// Sparse-support MCMC fitter: 6-sigma ROI, FP32 persistent sums, warp reductions.
#define main reference_benchmark_main
#include "desk_benchmark.cu"
#undef main
#include <cstdio>
struct Packed {float x,y,sx,sy,angle,a,co,si,ix,iy,rx,ry;};
HD Packed packed(Gaussian<float> a){Packed g;g.x=a.v[0];g.y=a.v[1];g.sx=a.v[2];g.sy=a.v[3];g.angle=a.v[4];g.a=a.v[5];g.co=cosf(-g.angle);g.si=sinf(-g.angle);g.ix=1/g.sx;g.iy=1/g.sy;g.rx=6*sqrtf(g.sx*g.sx*g.co*g.co+g.sy*g.sy*g.si*g.si);g.ry=6*sqrtf(g.sx*g.sx*g.si*g.si+g.sy*g.sy*g.co*g.co);return g;}
HD float sparseValue(Packed g,int x,int y){float dx=x-g.x,dy=y-g.y,u=(dx*g.co-dy*g.si)*g.ix,v=(dx*g.si+dy*g.co)*g.iy,q=u*u+v*v;return q>36?0:g.a*expf(-.5f*q);}
constexpr int THREADS=128;
__global__ void sparseInit(Packed* gs,float* sum,const unsigned char* target,int W,int H,int N){
 int p=blockIdx.x*blockDim.x+threadIdx.x,c=blockIdx.y,P=W*H;if(p>=P)return;float s=0;
 for(int j=0;j<N;j++)s+=sparseValue(gs[c*N+j],p%W,p/W);sum[c*P+p]=s;
}
__global__ void sparseLoss(float* sum,long long* sse,const unsigned char* target,int P){
 int c=blockIdx.x,tid=threadIdx.x;long long s=0;for(int p=tid;p<P;p+=THREADS){long long d=quantize(sum[c*P+p])-target[p];s+=d*d;}
 __shared__ long long partial[4];for(int d=16;d;d/=2)s+=__shfl_down_sync(0xffffffff,s,d);if(tid%32==0)partial[tid/32]=s;__syncthreads();
 if(tid==0)sse[c]=partial[0]+partial[1]+partial[2]+partial[3];
}
__global__ void sparseIterate(Packed* gs,float* sum,float* deltas,long long* sse,uint32_t* rngs,int* acceptCounts,const unsigned char* target,int W,int H,int N,int batch,int offset,int steps,float T0,float T1,unsigned long long* pixels){
 int c=blockIdx.x,tid=threadIdx.x,P=W*H,base=c*P;
 __shared__ Packed oldg,newg;__shared__ int j,x0,y0,rw,rh,yes;__shared__ float uniformU,temp;__shared__ long long partial[4];
 for(int it=0;it<batch;it++){
  if(tid==0){
   uint32_t r=rngs[c];j=int(uniform(r)*N);int k=int(uniform(r)*6);float change=float((uniform(r)-.5)*2);uniformU=float(uniform(r));rngs[c]=r;
   float ratio=float(offset+it)/max(1,steps-1),factor=.15f+.85f*powf(T1/T0,ratio*.25f);temp=T0*powf(T1/T0,ratio);
   oldg=gs[c*N+j];Gaussian<float> a{{oldg.x,oldg.y,oldg.sx,oldg.sy,oldg.angle,oldg.a}};float step=k<2?2:(k<4?1:(k==4?.1f:10));a.v[k]+=change*step*factor;
   a.v[0]=bound(a.v[0],0.f,float(W));a.v[1]=bound(a.v[1],0.f,float(H));a.v[2]=fmaxf(.1f,a.v[2]);a.v[3]=fmaxf(.1f,a.v[3]);a.v[5]=bound(a.v[5],0.f,255.f);newg=packed(a);
   x0=max(0,int(floorf(fminf(oldg.x-oldg.rx,newg.x-newg.rx))));y0=max(0,int(floorf(fminf(oldg.y-oldg.ry,newg.y-newg.ry))));
   int x1=min(W-1,int(ceilf(fmaxf(oldg.x+oldg.rx,newg.x+newg.rx)))),y1=min(H-1,int(ceilf(fmaxf(oldg.y+oldg.ry,newg.y+newg.ry))));
   rw=max(0,x1-x0+1);rh=max(0,y1-y0+1);pixels[c]+=rw*rh;
  }
  __syncthreads();long long diff=0;
  for(int q=tid;q<rw*rh;q+=THREADS){int x=x0+q%rw,y=y0+q/rw,p=y*W+x;float delta=sparseValue(newg,x,y)-sparseValue(oldg,x,y);deltas[base+p]=delta;
   long long before=quantize(sum[base+p])-target[p],after=quantize(sum[base+p]+delta)-target[p];diff+=after*after-before*before;}
  for(int d=16;d;d/=2)diff+=__shfl_down_sync(0xffffffff,diff,d);if(tid%32==0)partial[tid/32]=diff;__syncthreads();
  if(tid==0){long long delta=partial[0]+partial[1]+partial[2]+partial[3];yes=uniformU<fminf(1.f,expf(-float(delta)/P/temp));if(yes){sse[c]+=delta;gs[c*N+j]=newg;acceptCounts[c]++;}}
  __syncthreads();if(yes){for(int q=tid;q<rw*rh;q+=THREADS){int p=(y0+q/rw)*W+x0+q%rw;sum[base+p]+=deltas[base+p];}}__syncthreads();
 }
}
int main(int argc,char** argv){try{
 auto entire=Clock::now();std::string input="zhou_target.pgm",out="results/roi";int N=800,C=8,steps=1000000;float T0=2,T1=.0005;
 for(int i=1;i<argc;i++){std::string a=argv[i];if(i+1>=argc)throw std::runtime_error("missing value");std::string v=argv[++i];if(a=="--input")input=v;else if(a=="--output")out=v;else if(a=="--gaussians")N=std::stoi(v);else if(a=="--chains")C=std::stoi(v);else if(a=="--steps")steps=std::stoi(v);else if(a=="--initial-temperature")T0=std::stof(v);else if(a=="--final-temperature")T1=std::stof(v);else throw std::runtime_error("unknown option");}
 if(N<1||N>4096||C<1||C>128||steps<1||T0<=0||T1<=0||!std::isfinite(T0)||!std::isfinite(T1))throw std::runtime_error("invalid configuration");
 int W=0,H=0;auto tar=readPgm(input,W,H);int P=W*H;std::vector<double> cdf(P);double mass=0;for(int p=0;p<P;p++){mass+=tar[p];cdf[p]=mass;}if(mass==0)throw std::runtime_error("empty target");
 std::vector<Packed> gs(C*N);std::vector<uint32_t> seeds(C);for(int c=0;c<C;c++){uint32_t r=42+c*1009;seeds[c]=101+c*1013;for(int j=0;j<N;j++){
  int p=std::lower_bound(cdf.begin(),cdf.end(),(j+uniform(r))*mass/N)-cdf.begin();float sx=1.1f+uniform(r)*1.5f,sy=1.1f+uniform(r)*1.5f,angle=uniform(r)*6.283185307179586;
  Gaussian<float> g{{float(p%W),float(p/W),sx,sy,angle,float(std::min(230.,mass/(N*6.283185307179586*sx*sy)))}};gs[c*N+j]=packed(g);
 }}
 size_t bytes=size_t(C)*P*8+gs.size()*sizeof(Packed)+P+C*24;size_t free,total;CK(cudaMemGetInfo(&free,&total));if(bytes>free/2)throw std::runtime_error("VRAM budget exceeded");
 auto dg=alloc<Packed>(gs.size());auto sum=alloc<float>(C*P);auto delta=alloc<float>(C*P);auto dt=alloc<unsigned char>(P);auto dsse=alloc<long long>(C);auto drng=alloc<uint32_t>(C);auto da=alloc<int>(C);auto dp=alloc<unsigned long long>(C);
 CK(cudaMemcpy(dg,gs.data(),gs.size()*sizeof(Packed),cudaMemcpyHostToDevice));CK(cudaMemcpy(dt,tar.data(),P,cudaMemcpyHostToDevice));CK(cudaMemcpy(drng,seeds.data(),C*4,cudaMemcpyHostToDevice));CK(cudaMemset(da,0,C*4));CK(cudaMemset(dp,0,C*8));
 sparseInit<<<dim3((P+THREADS-1)/THREADS,C),THREADS>>>(dg,sum,dt,W,H,N);sparseLoss<<<C,THREADS>>>(sum,dsse,dt,P);CK(cudaGetLastError());CK(cudaDeviceSynchronize());double setup=elapsed(entire);
 cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));CK(cudaEventRecord(begin));
 for(int o=0;o<steps;o+=1024){sparseIterate<<<C,THREADS>>>(dg,sum,delta,dsse,drng,da,dt,W,H,N,std::min(1024,steps-o),o,steps,T0,T1,dp);CK(cudaGetLastError());}
 CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));
 std::vector<long long> losses(C);std::vector<int> counts(C);std::vector<unsigned long long> visited(C);CK(cudaMemcpy(losses.data(),dsse,C*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(counts.data(),da,C*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(visited.data(),dp,C*8,cudaMemcpyDeviceToHost));CK(cudaMemcpy(gs.data(),dg,gs.size()*sizeof(Packed),cudaMemcpyDeviceToHost));
 int winner=std::min_element(losses.begin(),losses.end())-losses.begin();std::vector<float> cached(P);CK(cudaMemcpy(cached.data(),sum+winner*P,P*4,cudaMemcpyDeviceToHost));
 // Independent full, untruncated FP64 evaluation of the selected final parameters.
 std::vector<double> exact(P,0);for(int j=0;j<N;j++){auto g=gs[winner*N+j];Gaussian<double> ref{{g.x,g.y,g.sx,g.sy,g.angle,g.a}};double co=cos(-ref.v[4]),si=sin(-ref.v[4]);for(int p=0;p<P;p++)exact[p]+=value(ref,p%W,p/W,co,si);}
 std::vector<unsigned char> image(P);long long fullSse=0,cachedSse=0;double maxDrift=0,pixelMae=0;int pixelMax=0;
 for(int p=0;p<P;p++){image[p]=quantize(exact[p]);int cv=quantize(cached[p]);long long d=int(image[p])-tar[p];fullSse+=d*d;d=cv-tar[p];cachedSse+=d*d;maxDrift=std::max(maxDrift,std::abs(exact[p]-cached[p]));int e=abs(int(image[p])-cv);pixelMae+=e;pixelMax=std::max(pixelMax,e);}
 if(cachedSse!=losses[winner])throw std::runtime_error("ROI delta-SSE invariant failed");
 std::ofstream pgm(out+".pgm",std::ios::binary);if(!pgm)throw std::runtime_error("cannot write PGM");pgm<<"P5\n"<<W<<" "<<H<<"\n255\n";pgm.write((char*)image.data(),P);pgm.close();
 unsigned long long visits=0;for(auto v:visited)visits+=v;double endToEnd=elapsed(entire),loss=double(fullSse)/P;
 std::ofstream json(out+".json");if(!json)throw std::runtime_error("cannot write JSON");json<<std::setprecision(12)<<"{\"W\":"<<W<<",\"H\":"<<H<<",\"N\":"<<N<<",\"chains\":"<<C<<",\"steps\":"<<steps<<",\"dtype\":\"fp32\",\"roi_sigma\":6,\"initial_T\":"<<T0<<",\"final_T\":"<<T1<<",\"setup_ms\":"<<setup<<",\"gpu_ms\":"<<ms<<",\"end_to_end_ms\":"<<endToEnd<<",\"loss\":"<<loss<<",\"psnr\":"<<10*log10(255.*255/loss)<<",\"cached_loss\":"<<double(losses[winner])/P<<",\"sse_invariant\":true,\"max_sum_drift\":"<<maxDrift<<",\"pixel_mae\":"<<pixelMae/P<<",\"pixel_max_diff\":"<<pixelMax<<",\"workspace_bytes\":"<<bytes<<",\"avg_roi_pixels\":"<<double(visits)/(C*steps)<<",\"winner\":"<<winner<<",\"accepted\":"<<counts[winner]<<",\"gaussians\":[";
 for(int j=0;j<N;j++){if(j)json<<",";auto g=gs[winner*N+j];json<<"["<<g.x<<","<<g.y<<","<<g.sx<<","<<g.sy<<","<<g.angle<<","<<g.a<<"]";}json<<"]}\n";
 std::cout<<"ROI "<<steps<<" steps x "<<C<<" chains: GPU "<<ms<<" ms, end-to-end "<<endToEnd<<" ms, full FP64 MSE "<<loss<<", ROI "<<double(visits)/(C*steps)<<" pixels\n";
 for(void* p:{(void*)dg,(void*)sum,(void*)delta,(void*)dt,(void*)dsse,(void*)drng,(void*)da,(void*)dp})CK(cudaFree(p));CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}


// Standalone CUDA + CPU benchmark. No Python/PyTorch dependency.
// nvcc -O3 -std=c++17 -arch=sm_75 --fmad=false src/desk_benchmark.cu -o desk_benchmark
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#define HD __host__ __device__
#define CK(call) do{cudaError_t e=(call);if(e!=cudaSuccess)throw std::runtime_error(std::string(#call)+": "+cudaGetErrorString(e));}while(0)
constexpr int BLOCK=256;
template<class R> struct Gaussian {R v[6];};
HD uint32_t next(uint32_t& s){s^=s<<13;s^=s>>17;s^=s<<5;return s;}
HD double uniform(uint32_t& s){return double(next(s))/4294967296.;}
template<class R> HD R bound(R x,R lo,R hi){return x<lo?lo:(x>hi?hi:x);}
template<class R> HD int quantize(R s){return int(bound(floor(s+R(.5)),R(0),R(255)));}
template<class R> HD R value(const Gaussian<R>& g,int x,int y,R c,R s){
  R dx=R(x)-g.v[0],dy=R(y)-g.v[1];
  R a=(dx*c-dy*s)/g.v[2],b=(dx*s+dy*c)/g.v[3];return g.v[5]*exp(R(-.5)*(a*a+b*b));
}
template<class R> HD Gaussian<R> proposal(const Gaussian<R>& old,int param,double u,int W,int H,double factor=1){
  Gaussian<R> g=old;double step=param<2?2:(param<4?1:(param==4?.1:10));
  g.v[param]+=R((u-.5)*2*step*factor);g.v[0]=bound(g.v[0],R(0),R(W));g.v[1]=bound(g.v[1],R(0),R(H));
  g.v[2]=g.v[2]<R(.1)?R(.1):g.v[2];g.v[3]=g.v[3]<R(.1)?R(.1):g.v[3];g.v[5]=bound(g.v[5],R(0),R(255));return g;
}
HD bool accept(long long candidate,long long current,int P,double T,double u){return u<fmin(1.,exp(-(double(candidate)/P-double(current)/P)/T));}
template<class R> std::vector<Gaussian<R>> initial(int W,int H,int N,int C){
  std::vector<Gaussian<R>> gs(size_t(N)*C);
  for(int c=0;c<C;c++){uint32_t s=42+c*1009;for(int j=0;j<N;j++){
    auto& g=gs[c*N+j];g.v[0]=R(uniform(s)*W);g.v[1]=R(uniform(s)*H);
    g.v[2]=R(uniform(s)*W/10+double(W)/20);g.v[3]=R(uniform(s)*H/10+double(H)/20);
    g.v[4]=R(uniform(s)*2*3.141592653589793);g.v[5]=R(uniform(s)*100+50);
  }}return gs;
}
std::vector<unsigned char> target(int W,int H){
  const char* glyphs[2]={"10001110111010110101100011000110001","01111100001000010000100001000001111"};
  std::vector<unsigned char> a(W*H);double sc=std::min(double(W)/27,double(H)/11),ox=(W-23*sc)/2,oy=(H-7*sc)/2;
  for(int y=0;y<H;y++)for(int x=0;x<W;x++){
    int v=0;for(int sy=0;sy<2;sy++)for(int sx=0;sx<2;sx++){
      int gx=int(floor((x+(sx+.5)/2-ox)/sc)),gy=int(floor((y+(sy+.5)/2-oy)/sc));
      if(gx>=0&&gx<23&&gy>=0&&gy<7&&gx%6<5&&glyphs[(gx/6)%2][gy*5+gx%6]=='1')v++;
    }a[y*W+x]=int(floor(v*255./4+.5));
  }return a;
}
template<class R> struct CpuChain {
  int W,H,N,P;std::vector<Gaussian<R>> gs;std::vector<R> sum,cache,candidate;const std::vector<unsigned char>& t;
  uint32_t state;long long sse=0;int accepted=0;
  CpuChain(const std::vector<Gaussian<R>>& g,int w,int h,const std::vector<unsigned char>& tar,uint32_t seed):W(w),H(h),N(g.size()),P(w*h),gs(g),sum(P),cache(size_t(P)*N),candidate(P),t(tar),state(seed){
    for(int j=0;j<N;j++){R c=cos(gs[j].v[4]),s=sin(-gs[j].v[4]);for(int p=0;p<P;p++){R v=value(gs[j],p%W,p/W,c,s);cache[size_t(j)*P+p]=v;sum[p]+=v;}}
    for(int p=0;p<P;p++){long long d=quantize(sum[p])-t[p];sse+=d*d;}
  }
  bool step(bool full,double T=50){
    int j=int(uniform(state)*N),k=int(uniform(state)*6);auto g=proposal(gs[j],k,uniform(state),W,H);double u=uniform(state);
    long long candidateSse=0;R c=cos(g.v[4]),s=sin(-g.v[4]);
    for(int p=0;p<P;p++){
      R total=0;
      if(full){for(int z=0;z<N;z++){const auto& a=z==j?g:gs[z];total+=value(a,p%W,p/W,R(cos(a.v[4])),R(sin(-a.v[4])));}}
      else{candidate[p]=value(g,p%W,p/W,c,s);total=sum[p]-cache[size_t(j)*P+p]+candidate[p];}
      long long d=quantize(total)-t[p];candidateSse+=d*d;
    }
    bool yes=accept(candidateSse,sse,P,T,u);
    if(yes){for(int p=0;p<P;p++){R v=full?value(g,p%W,p/W,c,s):candidate[p];sum[p]=sum[p]-cache[size_t(j)*P+p]+v;cache[size_t(j)*P+p]=v;}gs[j]=g;sse=candidateSse;accepted++;}return yes;
  }
};
template<class R> __global__ void initKernel(Gaussian<R>* gs,R* sums,R* cache,long long* sses,int* accepted,uint32_t* states,const unsigned char* target,int W,int H,int N,bool reset=true){
  int chain=blockIdx.x,P=W*H,tid=threadIdx.x;long long local=0;__shared__ long long reduce[BLOCK];
  for(int p=tid;p<P;p+=BLOCK){R total=0;for(int j=0;j<N;j++){auto g=gs[chain*N+j];R v=value(g,p%W,p/W,R(cos(g.v[4])),R(sin(-g.v[4])));cache[(size_t(chain)*N+j)*P+p]=v;total+=v;}sums[size_t(chain)*P+p]=total;long long d=quantize(total)-target[p];local+=d*d;}
  reduce[tid]=local;__syncthreads();for(int stride=BLOCK/2;stride;stride/=2){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
  if(tid==0){sses[chain]=reduce[0];if(reset){accepted[chain]=0;states[chain]=101+chain*1013;}}
}
template<class R> __global__ void iterateKernel(Gaussian<R>* gs,R* sums,R* cache,R* candidate,long long* sses,int* accepted,uint32_t* states,const unsigned char* target,int W,int H,int N,int steps,double T,long long* trace,int* decisions,int offset=0,int totalSteps=1,double finalT=50,bool annealProposal=false){
  int chain=blockIdx.x,P=W*H,tid=threadIdx.x;size_t base=size_t(chain)*P;
  __shared__ Gaussian<R> g;__shared__ int index,yes;__shared__ double u;__shared__ R co,si;
  __shared__ long long reduce[BLOCK];
  for(int it=0;it<steps;it++){
    if(tid==0){uint32_t st=states[chain];index=int(uniform(st)*N);int k=int(uniform(st)*6);double factor=annealProposal?.15+.85*pow(finalT/T,double(offset+it)/(totalSteps>1?totalSteps-1:1)*.25):1;g=proposal(gs[chain*N+index],k,uniform(st),W,H,factor);u=uniform(st);states[chain]=st;co=cos(g.v[4]);si=sin(-g.v[4]);}
    __syncthreads();long long local=0;size_t old=(size_t(chain)*N+index)*P;
    for(int p=tid;p<P;p+=BLOCK){R v=value(g,p%W,p/W,co,si);candidate[base+p]=v;R total=sums[base+p]-cache[old+p]+v;long long d=quantize(total)-target[p];local+=d*d;}
    reduce[tid]=local;__syncthreads();for(int stride=BLOCK/2;stride;stride/=2){if(tid<stride)reduce[tid]+=reduce[tid+stride];__syncthreads();}
    if(tid==0){double temp=T==finalT?T:T*pow(finalT/T,double(offset+it)/(totalSteps>1?totalSteps-1:1));yes=accept(reduce[0],sses[chain],P,temp,u);if(yes){sses[chain]=reduce[0];accepted[chain]++;gs[chain*N+index]=g;}if(trace&&chain==0){trace[it]=sses[chain];decisions[it]=yes;}}
    __syncthreads();if(yes){for(int p=tid;p<P;p+=BLOCK){R v=candidate[base+p];sums[base+p]=sums[base+p]-cache[old+p]+v;cache[old+p]=v;}}
    __syncthreads(); // No proposal observes a partially committed image.
  }
}
using Clock=std::chrono::steady_clock;
double elapsed(Clock::time_point a){return std::chrono::duration<double,std::milli>(Clock::now()-a).count();}
template<class T> T* alloc(size_t n){T* p;CK(cudaMalloc(&p,n*sizeof(T)));return p;}
std::vector<unsigned char> readPgm(const std::string& file,int& W,int& H){
  std::ifstream in(file,std::ios::binary);if(!in)throw std::runtime_error("Cannot open input PGM");
  auto token=[&](){std::string t;while(in>>t){if(t[0]=='#'){std::getline(in,t);continue;}return t;}throw std::runtime_error("Invalid PGM header");};
  if(token()!="P5")throw std::runtime_error("Input must be binary grayscale PGM (P5)");W=std::stoi(token());H=std::stoi(token());if(std::stoi(token())!=255||W<1||H<1||W>2048||H>2048)throw std::runtime_error("PGM dimensions/maxval unsupported");
  char c=in.get();if(c=='\r'&&in.peek()=='\n')in.get();std::vector<unsigned char> data(W*H);in.read((char*)data.data(),data.size());if(size_t(in.gcount())!=data.size())throw std::runtime_error("Truncated PGM");return data;
}
template<class R> void fit(int W,int H,int N,int C,int steps,const std::string& input,const std::string& output,double finalT,double initialT=50,bool seeded=false){
  auto tar=input.empty()?target(W,H):readPgm(input,W,H);int P=W*H;size_t free,total;CK(cudaMemGetInfo(&free,&total));
  size_t bytes=size_t(C)*P*(N+2)*sizeof(R)+size_t(C)*N*sizeof(Gaussian<R>)+P+C*(sizeof(long long)+sizeof(int)+sizeof(uint32_t));
  if(bytes>free/2)throw std::runtime_error("Workspace exceeds half of free VRAM");auto gs=initial<R>(W,H,N,C);
  if(seeded){std::vector<double> cdf(P);double mass=0;for(int p=0;p<P;p++){mass+=tar[p];cdf[p]=mass;}if(mass<=0)throw std::runtime_error("Target has no positive intensity");for(int c=0;c<C;c++){uint32_t st=42+c*1009;for(int j=0;j<N;j++){int p=int(std::lower_bound(cdf.begin(),cdf.end(),(j+uniform(st))*mass/N)-cdf.begin());auto& g=gs[c*N+j];g.v[0]=R(p%W);g.v[1]=R(p/W);g.v[2]=R(1.1+uniform(st)*1.5);g.v[3]=R(1.1+uniform(st)*1.5);g.v[4]=R(uniform(st)*6.283185307179586);g.v[5]=R(std::min(230.,mass/(N*6.283185307179586*double(g.v[2])*double(g.v[3]))));}}}
  auto dg=alloc<Gaussian<R>>(gs.size());auto sums=alloc<R>(size_t(C)*P);auto cache=alloc<R>(size_t(C)*N*P);auto candidate=alloc<R>(size_t(C)*P);auto sses=alloc<long long>(C);auto accepted=alloc<int>(C);auto states=alloc<uint32_t>(C);auto dt=alloc<unsigned char>(P);
  CK(cudaMemcpy(dt,tar.data(),P,cudaMemcpyHostToDevice));CK(cudaMemcpy(dg,gs.data(),gs.size()*sizeof(Gaussian<R>),cudaMemcpyHostToDevice));initKernel<<<C,BLOCK>>>(dg,sums,cache,sses,accepted,states,dt,W,H,N);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  cudaEvent_t a,b;CK(cudaEventCreate(&a));CK(cudaEventCreate(&b));auto start=Clock::now();CK(cudaEventRecord(a));
  for(int offset=0;offset<steps;offset+=256){if(seeded&&offset>0&&offset%8192==0){initKernel<<<C,BLOCK>>>(dg,sums,cache,sses,accepted,states,dt,W,H,N,false);CK(cudaGetLastError());}iterateKernel<<<C,BLOCK>>>(dg,sums,cache,candidate,sses,accepted,states,dt,W,H,N,std::min(256,steps-offset),initialT,nullptr,nullptr,offset,steps,finalT,seeded);CK(cudaGetLastError());}
  CK(cudaEventRecord(b));CK(cudaEventSynchronize(b));float ms;CK(cudaEventElapsedTime(&ms,a,b));std::vector<long long> losses(C);CK(cudaMemcpy(losses.data(),sses,C*sizeof(long long),cudaMemcpyDeviceToHost));CK(cudaMemcpy(gs.data(),dg,gs.size()*sizeof(Gaussian<R>),cudaMemcpyDeviceToHost));double wall=elapsed(start);
  // Re-render each final chain in FP64 before selecting the winner; never trust a drifted cache for export.
  long long best=INT64_MAX;int winner=-1;double maxDrift=0;std::vector<unsigned char> image(P);
  for(int c=0;c<C;c++){std::vector<Gaussian<double>> finalGs(N);for(int j=0;j<N;j++)for(int k=0;k<6;k++)finalGs[j].v[k]=double(gs[c*N+j].v[k]);CpuChain<double> verify(finalGs,W,H,tar,101);maxDrift=std::max(maxDrift,std::abs(double(verify.sse-losses[c])/P));if(verify.sse<best){best=verify.sse;winner=c;for(int p=0;p<P;p++)image[p]=quantize(verify.sum[p]);}}
  std::ofstream pgm(output+".pgm",std::ios::binary);if(!pgm)throw std::runtime_error("Cannot write output PGM");pgm<<"P5\n"<<W<<" "<<H<<"\n255\n";pgm.write((char*)image.data(),P);
  std::ofstream json(output+".json");if(!json)throw std::runtime_error("Cannot write output JSON");json<<std::setprecision(12)<<"{\"W\":"<<W<<",\"H\":"<<H<<",\"N\":"<<N<<",\"chains\":"<<C<<",\"steps\":"<<steps<<",\"dtype\":\""<<(sizeof(R)==8?"fp64":"fp32")<<"\",\"initial_T\":"<<initialT<<",\"seeded\":"<<(seeded?"true":"false")<<",\"final_T\":"<<finalT<<",\"gpu_ms\":"<<ms<<",\"wall_ms\":"<<wall<<",\"workspace_bytes\":"<<bytes<<",\"best_chain\":"<<winner<<",\"loss\":"<<double(best)/P<<",\"max_final_rerender_mse_drift\":"<<maxDrift<<",\"gaussians\":[";
  for(int j=0;j<N;j++){if(j)json<<",";json<<"[";for(int k=0;k<6;k++){if(k)json<<",";json<<gs[winner*N+j].v[k];}json<<"]";}json<<"]}\n";
  std::cout<<"fit: "<<C<<" chains x "<<steps<<" steps, "<<ms<<" ms, best final MSE="<<double(best)/P<<"; wrote "<<output<<".{pgm,json}\n";
  for(void* p:{(void*)dg,(void*)sums,(void*)cache,(void*)candidate,(void*)sses,(void*)accepted,(void*)states,(void*)dt})CK(cudaFree(p));CK(cudaEventDestroy(a));CK(cudaEventDestroy(b));
}
template<class R> void run(int W,int H,int N,int C,int steps,int reps){
  const int P=W*H,verify=500;cudaDeviceProp prop;CK(cudaGetDeviceProperties(&prop,0));size_t free,total;CK(cudaMemGetInfo(&free,&total));
  size_t bytes=size_t(C)*P*(N+2)*sizeof(R)+size_t(C)*N*sizeof(Gaussian<R>)+P+C*(sizeof(long long)+sizeof(int)+sizeof(uint32_t));
  if(bytes>free/2)throw std::runtime_error("Workspace exceeds half of currently free VRAM; reduce chains/resolution/Gaussians");
  auto tar=target(W,H);auto gs=initial<R>(W,H,N,C);std::vector<Gaussian<R>> one(gs.begin(),gs.begin()+N);
  auto setupStart=Clock::now();auto dg=alloc<Gaussian<R>>(gs.size());auto sums=alloc<R>(size_t(C)*P);auto cache=alloc<R>(size_t(C)*N*P);auto candidate=alloc<R>(size_t(C)*P);
  auto sses=alloc<long long>(C);auto accepted=alloc<int>(C);auto states=alloc<uint32_t>(C);auto dt=alloc<unsigned char>(P);auto trace=alloc<long long>(verify);auto decisions=alloc<int>(verify);
  CK(cudaMemcpy(dt,tar.data(),P,cudaMemcpyHostToDevice));
  auto reset=[&](){CK(cudaMemcpy(dg,gs.data(),gs.size()*sizeof(Gaussian<R>),cudaMemcpyHostToDevice));initKernel<<<C,BLOCK>>>(dg,sums,cache,sses,accepted,states,dt,W,H,N);CK(cudaGetLastError());CK(cudaDeviceSynchronize());};
  reset();double setup_ms=elapsed(setupStart);
  // Check every decision and integer SSE against an independently fully rendered CPU chain.
  iterateKernel<<<C,BLOCK>>>(dg,sums,cache,candidate,sses,accepted,states,dt,W,H,N,verify,50.,trace,decisions);CK(cudaGetLastError());CK(cudaDeviceSynchronize());
  std::vector<long long> tr(verify);std::vector<int> de(verify);CK(cudaMemcpy(tr.data(),trace,verify*sizeof(long long),cudaMemcpyDeviceToHost));CK(cudaMemcpy(de.data(),decisions,verify*sizeof(int),cudaMemcpyDeviceToHost));
  CpuChain<R> reference(one,W,H,tar,101),cached(one,W,H,tar,101);int decision_mismatch=0,loss_mismatch=0,cpu_mismatch=0;long long maxSseError=0;
  for(int i=0;i<verify;i++){bool a=reference.step(true),b=cached.step(false);decision_mismatch+=int(a)!=de[i];loss_mismatch+=reference.sse!=tr[i];cpu_mismatch+=(a!=b||reference.sse!=cached.sse);maxSseError=std::max(maxSseError,llabs(reference.sse-tr[i]));}
  auto canonicalGs=initial<double>(W,H,N,1);CpuChain<double> canonical(canonicalGs,W,H,tar,101);int canonical_decisions=0,canonical_losses=0;double canonical_max_error=0;
  for(int i=0;i<verify;i++){bool yes=canonical.step(true);canonical_decisions+=int(yes)!=de[i];canonical_losses+=canonical.sse!=tr[i];canonical_max_error=std::max(canonical_max_error,std::abs(double(canonical.sse-tr[i])/P));}
  std::vector<double> fullTimes,cachedTimes,gpuTimes,wallTimes;std::vector<long long> lastSse(C);std::vector<int> lastAccepted(C);int final_mismatch=0;
  cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));
  for(int rep=0;rep<reps;rep++){
    CpuChain<R> full(one,W,H,tar,101);auto start=Clock::now();for(int i=0;i<steps;i++)full.step(true);fullTimes.push_back(elapsed(start));
    std::vector<CpuChain<R>> cpus;cpus.reserve(C);for(int c=0;c<C;c++)cpus.emplace_back(std::vector<Gaussian<R>>(gs.begin()+c*N,gs.begin()+(c+1)*N),W,H,tar,101+c*1013);
    start=Clock::now();for(auto& cpu:cpus)for(int i=0;i<steps;i++)cpu.step(false);cachedTimes.push_back(elapsed(start));
    reset();start=Clock::now();CK(cudaEventRecord(begin));
    for(int offset=0;offset<steps;offset+=256){int count=std::min(256,steps-offset);iterateKernel<<<C,BLOCK>>>(dg,sums,cache,candidate,sses,accepted,states,dt,W,H,N,count,50.,nullptr,nullptr);CK(cudaGetLastError());}
    CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float ms;CK(cudaEventElapsedTime(&ms,begin,end));gpuTimes.push_back(ms);
    CK(cudaMemcpy(lastSse.data(),sses,C*sizeof(long long),cudaMemcpyDeviceToHost));CK(cudaMemcpy(lastAccepted.data(),accepted,C*sizeof(int),cudaMemcpyDeviceToHost));wallTimes.push_back(elapsed(start));
    if(rep==0)for(int c=0;c<C;c++)final_mismatch+=(lastSse[c]!=cpus[c].sse||lastAccepted[c]!=cpus[c].accepted);
    std::cerr<<"rep "<<rep<<" full CPU(single)="<<fullTimes.back()<<" cached CPU(all)="<<cachedTimes.back()<<" GPU(all)="<<ms<<" ms\n";
  }
  auto med=[](std::vector<double> a){std::sort(a.begin(),a.end());return a[a.size()/2];};
  auto array=[](const char* name,const std::vector<double>& v){std::cout<<"\""<<name<<"\":[";for(size_t i=0;i<v.size();i++){if(i)std::cout<<",";std::cout<<v[i];}std::cout<<"],\n";};
  std::cout<<std::setprecision(10)<<"{\n\"gpu\":\""<<prop.name<<"\",\"sm_count\":"<<prop.multiProcessorCount<<",\"dtype\":\""<<(sizeof(R)==8?"fp64":"fp32")<<"\",\n";
  std::cout<<"\"W\":"<<W<<",\"H\":"<<H<<",\"N\":"<<N<<",\"chains\":"<<C<<",\"steps\":"<<steps<<",\"repeats\":"<<reps<<",\"T\":50,\"chunk_steps\":256,\n";
  std::cout<<"\"workspace_bytes\":"<<bytes<<",\"free_vram_bytes_before\":"<<free<<",\"setup_ms\":"<<setup_ms<<",\n";
  std::cout<<"\"verification\":{\"steps\":"<<verify<<",\"decision_mismatch\":"<<decision_mismatch<<",\"loss_mismatch\":"<<loss_mismatch<<",\"cpu_mismatch\":"<<cpu_mismatch<<",\"max_mse_error\":"<<double(maxSseError)/P<<",\"final_chain_mismatch\":"<<final_mismatch<<",\"canonical_fp64_decision_mismatch\":"<<canonical_decisions<<",\"canonical_fp64_loss_mismatch\":"<<canonical_losses<<",\"canonical_fp64_max_mse_error\":"<<canonical_max_error<<"},\n";
  array("cpu_full_single_ms_samples",fullTimes);array("cpu_cached_all_ms_samples",cachedTimes);array("gpu_all_ms_samples",gpuTimes);array("gpu_wall_ms_samples",wallTimes);
  std::cout<<"\"cpu_full_single_ms\":"<<med(fullTimes)<<",\"cpu_cached_all_ms\":"<<med(cachedTimes)<<",\"gpu_all_ms\":"<<med(gpuTimes)<<",\"gpu_wall_ms\":"<<med(wallTimes)<<",\n";
  std::cout<<"\"gpu_total_steps_s\":"<<double(C)*steps*1000/med(gpuTimes)<<",\"gpu_vs_cpu_cached_same_work\":"<<med(cachedTimes)/med(gpuTimes)<<",\"chain0_loss\":"<<double(lastSse[0])/P<<",\"chain0_accepted\":"<<lastAccepted[0]<<"\n}\n";
  for(void* p:{(void*)dg,(void*)sums,(void*)cache,(void*)candidate,(void*)sses,(void*)accepted,(void*)states,(void*)dt,(void*)trace,(void*)decisions})CK(cudaFree(p));CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));
  if(sizeof(R)==8&&(decision_mismatch||loss_mismatch||cpu_mismatch||final_mismatch))throw std::runtime_error("FP64 validation failed; results are not validated speedups");
}
int main(int argc,char** argv){try{
  int W=100,H=50,N=25,C=1,steps=1200,reps=3;bool fp32=false,doFit=false,seeded=false;std::string input,output="fit";double finalT=.05,initialT=50;
  for(int i=1;i<argc;i++){std::string a=argv[i];if(a=="--fp32")fp32=true;else if(a=="--fit")doFit=true;else if(a=="--seeded")seeded=true;else{if(i+1>=argc)throw std::runtime_error("missing option value");std::string val=argv[++i];if(a=="--input")input=val;else if(a=="--output")output=val;else if(a=="--final-temperature")finalT=std::stod(val);else if(a=="--initial-temperature")initialT=std::stod(val);else{int v=std::stoi(val);if(a=="--width")W=v;else if(a=="--height")H=v;else if(a=="--gaussians")N=v;else if(a=="--chains")C=v;else if(a=="--steps")steps=v;else if(a=="--repeats")reps=v;else throw std::runtime_error("unknown option "+a);}}}
  if(W<1||H<1||N<1||C<1||steps<1||reps<1||W>2048||H>2048||N>1024||C>1024)throw std::runtime_error("invalid size");
  if(!std::isfinite(finalT)||finalT<=0||!std::isfinite(initialT)||initialT<=0)throw std::runtime_error("temperatures must be finite and positive");
  if(doFit){if(fp32)fit<float>(W,H,N,C,steps,input,output,finalT,initialT,seeded);else fit<double>(W,H,N,C,steps,input,output,finalT,initialT,seeded);}else{if(!input.empty())throw std::runtime_error("--input requires --fit");if(fp32)run<float>(W,H,N,C,steps,reps);else run<double>(W,H,N,C,steps,reps);}return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

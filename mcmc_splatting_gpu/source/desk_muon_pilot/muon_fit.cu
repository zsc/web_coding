// Experimental Muon adaptation for Gaussian parameters. FP32 NS5, zero weight decay.
// Source coefficients/momentum: source/muon_upstream.py, pinned provenance alongside it.
#define ROI_LIBRARY_ONLY
#include "roi_fit.cu"
__global__ void refineAdamScaled(Packed* gs,const float* sum,const unsigned char* target,float* m,float* v,int W,int H,int N,int step,int total,float lr){
 int j=blockIdx.x,c=blockIdx.y,tid=threadIdx.x,index=c*N+j;Packed g=gs[index];
 int x0=max(0,int(floorf(g.x-g.rx))),y0=max(0,int(floorf(g.y-g.ry))),x1=min(W-1,int(ceilf(g.x+g.rx))),y1=min(H-1,int(ceilf(g.y+g.ry))),rw=max(0,x1-x0+1),rh=max(0,y1-y0+1);
 float grad[6]={0,0,0,0,0,0};
 for(int q=tid;q<rw*rh;q+=THREADS){
  int x=x0+q%rw,y=y0+q/rw,p=y*W+x;float dx=x-g.x,dy=y-g.y,u=dx*g.co-dy*g.si,w=dx*g.si+dy*g.co,ux=u*g.ix,wy=w*g.iy,qq=ux*ux+wy*wy;
  if(qq<=36){float e=expf(-.5f*qq),f=g.a*e,residual=sum[c*W*H+p]-target[p],d=2*residual*f,ix2=g.ix*g.ix,iy2=g.iy*g.iy;
   grad[0]+=d*(u*g.co*ix2+w*g.si*iy2);grad[1]+=d*(-u*g.si*ix2+w*g.co*iy2);grad[2]+=d*ux*ux;grad[3]+=d*wy*wy;grad[4]+=d*u*w*(iy2-ix2);grad[5]+=2*residual*e;
  }
 }
 __shared__ float partial[24];for(int k=0;k<6;k++){float a=grad[k];for(int s=16;s;s/=2)a+=__shfl_down_sync(0xffffffff,a,s);if(tid%32==0)partial[(tid/32)*6+k]=a;}__syncthreads();
 if(tid==0){
  float p[6]={g.x,g.y,logf(g.sx),logf(g.sy),g.angle,g.a};float base[6]={.15f,.15f,.025f,.025f,.025f,1.0f};
  float schedule=.05f+.95f*.5f*(1+cosf(3.141592653589793f*step/max(1,total-1)));
  float c1=1-powf(.9f,step+1),c2=1-powf(.999f,step+1);
  for(int k=0;k<6;k++){float gr=partial[k]+partial[6+k]+partial[12+k]+partial[18+k];int at=index*6+k;float mm=.9f*m[at]+.1f*gr,vv=.999f*v[at]+.001f*gr*gr;m[at]=mm;v[at]=vv;p[k]-=lr*base[k]*schedule*(mm/c1)/(sqrtf(vv/c2)+1e-8f);}
  Gaussian<float> a{{bound(p[0],0.f,float(W)),bound(p[1],0.f,float(H)),bound(expf(p[2]),.1f,32.f),bound(expf(p[3]),.1f,32.f),p[4],bound(p[5],0.f,255.f)}};gs[index]=packed(a);
 }
}
__global__ void muonGradient(Packed* gs,const float* sum,const unsigned char* target,float* m,float* v,int W,int H,int N,int step,int mode,float* update,float* audit){
 int j=blockIdx.x,c=blockIdx.y,tid=threadIdx.x,index=c*N+j;Packed g=gs[index];
 int x0=max(0,int(floorf(g.x-g.rx))),y0=max(0,int(floorf(g.y-g.ry))),x1=min(W-1,int(ceilf(g.x+g.rx))),y1=min(H-1,int(ceilf(g.y+g.ry))),rw=max(0,x1-x0+1),rh=max(0,y1-y0+1);
 float grad[6]={0,0,0,0,0,0};
 for(int q=tid;q<rw*rh;q+=THREADS){
  int x=x0+q%rw,y=y0+q/rw,p=y*W+x;float dx=x-g.x,dy=y-g.y,u=dx*g.co-dy*g.si,w=dx*g.si+dy*g.co,ux=u*g.ix,wy=w*g.iy,qq=ux*ux+wy*wy;
  if(qq<=36){float e=expf(-.5f*qq),f=g.a*e,residual=sum[c*W*H+p]-target[p],d=2*residual*f,ix2=g.ix*g.ix,iy2=g.iy*g.iy;
   grad[0]+=d*(u*g.co*ix2+w*g.si*iy2);grad[1]+=d*(-u*g.si*ix2+w*g.co*iy2);grad[2]+=d*ux*ux;grad[3]+=d*wy*wy;grad[4]+=d*u*w*(iy2-ix2);grad[5]+=2*residual*e;
  }
 }
 __shared__ float partial[24];for(int k=0;k<6;k++){float a=grad[k];for(int s=16;s;s/=2)a+=__shfl_down_sync(0xffffffff,a,s);if(tid%32==0)partial[(tid/32)*6+k]=a;}__syncthreads();
 if(tid==0){
  const float base[6]={.15f,.15f,.025f,.025f,.025f,1.f};
  for(int k=0;k<6;k++){
   float gr=partial[k]+partial[6+k]+partial[12+k]+partial[18+k];int at=index*6+k;
   if(audit) audit[at]=gr*base[k];
   if(mode==2 && k>=4){
    float mm=.9f*m[at]+.1f*gr,vv=.999f*v[at]+.001f*gr*gr;m[at]=mm;v[at]=vv;
    update[at]=(mm/(1-powf(.9f,step+1)))/(sqrtf(vv/(1-powf(.999f,step+1)))+1e-8f);
   }else{
    gr*=base[k];float mm=.95f*m[at]+.05f*gr;m[at]=mm;update[at]=.05f*gr+.95f*mm;
   }
  }
 }
}

// Each chain is independent; orthogonalize N x K, using only a K x K Gram.
// Shared X makes every row update read its own previous row before overwriting it.
template<int K> __global__ void orthogonalize(float* update,int N){
 const int c=blockIdx.y,offset=blockIdx.x*K,t=threadIdx.x;
 extern __shared__ float X[];
 __shared__ float partial[8*21],A[K*K],B[K*K],invnorm;
 float norm=0;
 for(int j=t;j<N;j+=256)for(int k=0;k<K;k++){float v=update[(c*N+j)*6+offset+k];X[j*K+k]=v;norm+=v*v;}
 for(int d=16;d;d/=2)norm+=__shfl_down_sync(0xffffffff,norm,d);
 if(t%32==0)partial[t/32]=norm;__syncthreads();
 if(t==0){float total=0;for(int i=0;i<8;i++)total+=partial[i];invnorm=1/(sqrtf(total)+1e-7f);}__syncthreads();
 for(int i=t;i<N*K;i+=256)X[i]*=invnorm;__syncthreads();
 for(int iteration=0;iteration<5;iteration++){
  float stats[K*(K+1)/2]={0};
  for(int j=t;j<N;j+=256){float row[K];for(int k=0;k<K;k++)row[k]=X[j*K+k];int at=0;for(int k=0;k<K;k++)for(int l=k;l<K;l++)stats[at++]+=row[k]*row[l];}
  for(int at=0;at<K*(K+1)/2;at++){float v=stats[at];for(int d=16;d;d/=2)v+=__shfl_down_sync(0xffffffff,v,d);if(t%32==0)partial[(t/32)*21+at]=v;}
  __syncthreads();
  if(t==0){int at=0;for(int k=0;k<K;k++)for(int l=k;l<K;l++){float v=0;for(int w=0;w<8;w++)v+=partial[w*21+at];A[k*K+l]=A[l*K+k]=v;at++;}}
  __syncthreads();
  if(t<K*K){int k=t/K,l=t%K;float aa=0;for(int m=0;m<K;m++)aa+=A[k*K+m]*A[m*K+l];B[t]=-4.7750f*A[t]+2.0315f*aa;}
  __syncthreads();
  for(int j=t;j<N;j+=256){float row[K],out[K];for(int k=0;k<K;k++)row[k]=X[j*K+k];for(int k=0;k<K;k++){float v=0;for(int l=0;l<K;l++)v+=B[k*K+l]*row[l];out[k]=3.4445f*row[k]+v;}for(int k=0;k<K;k++)X[j*K+k]=out[k];}
  __syncthreads();
 }
 float scale=sqrtf(fmaxf(1.f,float(N)/K));
 for(int j=t;j<N;j+=256)for(int k=0;k<K;k++)update[(c*N+j)*6+offset+k]=X[j*K+k]*scale;
}
__global__ void applyMuon(Packed* gs,const float* update,int W,int H,int N,int C,int step,int total,int mode,float lr){
 int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=N*C)return;Packed g=gs[i];
 float p[6]={g.x,g.y,logf(g.sx),logf(g.sy),g.angle,g.a};const float base[6]={.15f,.15f,.025f,.025f,.025f,1.f};
 float schedule=.05f+.95f*.5f*(1+cosf(3.141592653589793f*step/max(1,total-1)));
 for(int k=0;k<6;k++)p[k]-=base[k]*schedule*((mode==2&&k>=4)?1.f:lr)*update[i*6+k];
 gs[i]=packed(Gaussian<float>{{bound(p[0],0.f,float(W)),bound(p[1],0.f,float(H)),bound(expf(p[2]),.1f,32.f),bound(expf(p[3]),.1f,32.f),p[4],bound(p[5],0.f,255.f)}});
}
int main(int argc,char** argv){try{
 auto entire=Clock::now();std::string input="zhou_target.pgm",out="results/muon",auditPath;
 int N=1600,C=8,steps=8000,seed=42,mode=1;float lr=1;
 for(int i=1;i<argc;i++){std::string a=argv[i];if(i+1>=argc)throw std::runtime_error("missing value");std::string v=argv[++i];
  if(a=="--input")input=v;else if(a=="--output")out=v;else if(a=="--gaussians")N=std::stoi(v);else if(a=="--chains")C=std::stoi(v);
  else if(a=="--steps")steps=std::stoi(v);else if(a=="--mode")mode=std::stoi(v);else if(a=="--lr")lr=std::stof(v);
  else if(a=="--seed")seed=std::stoi(v);else if(a=="--audit")auditPath=v;else throw std::runtime_error("unknown option: "+a);
 }
 if(N<6||N>(mode==1?1900:4096)||C<1||C>128||steps<1||steps>1000000||mode<0||mode>2||!std::isfinite(lr)||lr<=0||(!auditPath.empty()&&mode==0))throw std::runtime_error("invalid configuration");
 int W=0,H=0;auto target=readPgm(input,W,H);int P=W*H;if(P>65536)throw std::runtime_error("image too large");
 std::vector<double> cdf(P);double mass=0;for(int p=0;p<P;p++)cdf[p]=(mass+=target[p]);if(mass==0)throw std::runtime_error("empty target");
 std::vector<Packed> gs(C*N);
 for(int c=0;c<C;c++){uint32_t r=uint32_t(seed+c*1009);for(int j=0;j<N;j++){
  int p=std::lower_bound(cdf.begin(),cdf.end(),(j+uniform(r))*mass/N)-cdf.begin();float sx=1.1f+uniform(r)*1.5f,sy=1.1f+uniform(r)*1.5f,angle=uniform(r)*6.283185307179586;
  gs[c*N+j]=packed(Gaussian<float>{{float(p%W),float(p/W),sx,sy,angle,float(std::min(230.,mass/(N*6.283185307179586*sx*sy)))}});
 }}
 auto initial=gs;size_t bytes=gs.size()*(sizeof(Packed)+18*sizeof(float))+size_t(C)*P*4+P+C*8+(auditPath.empty()?0:gs.size()*6*4);
 size_t free,total;CK(cudaMemGetInfo(&free,&total));if(bytes>free/2)throw std::runtime_error("VRAM budget exceeded");
 auto dg=alloc<Packed>(gs.size());auto sum=alloc<float>(C*P),moments=alloc<float>(gs.size()*12),update=alloc<float>(gs.size()*6);
 auto dt=alloc<unsigned char>(P);auto quantized=alloc<long long>(C);float* audit=auditPath.empty()?nullptr:alloc<float>(gs.size()*6);
 CK(cudaMemcpy(dg,gs.data(),gs.size()*sizeof(Packed),cudaMemcpyHostToDevice));CK(cudaMemcpy(dt,target.data(),P,cudaMemcpyHostToDevice));CK(cudaMemset(moments,0,gs.size()*48));
 CK(cudaMemset(sum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(dg,sum,W,H,N);CK(cudaGetLastError());CK(cudaDeviceSynchronize());double setup=elapsed(entire);
 cudaEvent_t begin,end;CK(cudaEventCreate(&begin));CK(cudaEventCreate(&end));CK(cudaEventRecord(begin));
 for(int a=0;a<steps;a++){
  CK(cudaMemsetAsync(sum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(dg,sum,W,H,N);
  if(mode==0)refineAdamScaled<<<dim3(N,C),THREADS>>>(dg,sum,dt,moments,moments+gs.size()*6,W,H,N,a,steps,lr);
  else{
   muonGradient<<<dim3(N,C),THREADS>>>(dg,sum,dt,moments,moments+gs.size()*6,W,H,N,a,mode,update,audit);
   if(mode==1)orthogonalize<6><<<dim3(1,C),256,N*6*4>>>(update,N);
   else orthogonalize<2><<<dim3(2,C),256,N*2*4>>>(update,N);
   applyMuon<<<(N*C+127)/128,128>>>(dg,update,W,H,N,C,a,steps,mode,lr);
  }
 }
 CK(cudaMemsetAsync(sum,0,C*P*4));renderSparse<<<dim3(N,C),THREADS>>>(dg,sum,W,H,N);sparseLoss<<<C,THREADS>>>(sum,quantized,dt,P);
 CK(cudaGetLastError());CK(cudaEventRecord(end));CK(cudaEventSynchronize(end));float gpuMs;CK(cudaEventElapsedTime(&gpuMs,begin,end));
 std::vector<long long> scores(C);CK(cudaMemcpy(gs.data(),dg,gs.size()*sizeof(Packed),cudaMemcpyDeviceToHost));CK(cudaMemcpy(scores.data(),quantized,C*8,cudaMemcpyDeviceToHost));
 int winner=std::min_element(scores.begin(),scores.end())-scores.begin();std::vector<float> cached(P);CK(cudaMemcpy(cached.data(),sum+winner*P,P*4,cudaMemcpyDeviceToHost));
 std::vector<double> full(P);for(int j=0;j<N;j++){auto g=gs[winner*N+j];Gaussian<double> ref{{g.x,g.y,g.sx,g.sy,g.angle,g.a}};double co=cos(-ref.v[4]),si=sin(-ref.v[4]);for(int p=0;p<P;p++)full[p]+=value(ref,p%W,p/W,co,si);}
 long long sse=0,cachedSSE=0;double smooth=0,pixelMae=0,maxSumDiff=0;std::vector<unsigned char> image(P);
 for(int p=0;p<P;p++){if(!std::isfinite(full[p]))throw std::runtime_error("nonfinite render");image[p]=quantize(full[p]);long long d=int(image[p])-target[p];sse+=d*d;d=quantize(cached[p])-target[p];cachedSSE+=d*d;smooth+=(full[p]-target[p])*(full[p]-target[p]);pixelMae+=abs(int(image[p])-quantize(cached[p]));maxSumDiff=std::max(maxSumDiff,abs(full[p]-cached[p]));}
 if(cachedSSE!=scores[winner])throw std::runtime_error("quantized loss mismatch");
 std::ofstream pgm(out+".pgm",std::ios::binary);if(!pgm)throw std::runtime_error("cannot write output");pgm<<"P5\n"<<W<<" "<<H<<"\n255\n";pgm.write((char*)image.data(),P);pgm.close();
 if(audit){std::vector<float> gr(gs.size()*6),updates(gr.size()),mom(gr.size());CK(cudaMemcpy(gr.data(),audit,gr.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(updates.data(),update,gr.size()*4,cudaMemcpyDeviceToHost));CK(cudaMemcpy(mom.data(),moments,gr.size()*4,cudaMemcpyDeviceToHost));std::ofstream f(auditPath);f<<std::setprecision(12)<<"{\"N\":"<<N<<",\"chains\":"<<C<<",\"mode\":"<<mode<<",\"steps\":"<<steps<<",\"scaled_gradient\":[";
  for(size_t i=0;i<gr.size();i++){if(i)f<<",";f<<gr[i];}f<<"],\"momentum\":[";for(size_t i=0;i<mom.size();i++){if(i)f<<",";f<<mom[i];}f<<"],\"update\":[";for(size_t i=0;i<updates.size();i++){if(i)f<<",";f<<updates[i];}f<<"]}\n";
 }
 std::ofstream json(out+".json");if(!json)throw std::runtime_error("cannot write JSON");json<<std::setprecision(12)<<"{\"backend\":\"muon_experiment\",\"dtype\":\"fp32\",\"W\":"<<W<<",\"H\":"<<H<<",\"N\":"<<N<<",\"chains\":"<<C<<",\"seed\":"<<seed<<",\"steps\":"<<steps<<",\"mode\":"<<mode<<",\"lr\":"<<lr<<",\"gpu_ms\":"<<gpuMs<<",\"setup_ms\":"<<setup<<",\"end_to_end_ms\":"<<elapsed(entire)<<",\"loss\":"<<double(sse)/P<<",\"smooth_loss\":"<<smooth/P<<",\"cached_loss\":"<<double(scores[winner])/P<<",\"pixel_mae\":"<<pixelMae/P<<",\"max_sum_drift\":"<<maxSumDiff<<",\"workspace_bytes\":"<<bytes<<",\"winner\":"<<winner<<",\"gaussians\":[";
 for(int j=0;j<N;j++){if(j)json<<",";auto g=gs[winner*N+j];json<<"["<<g.x<<","<<g.y<<","<<g.sx<<","<<g.sy<<","<<g.angle<<","<<g.a<<"]";}json<<"]}\n";json.close();
 std::cout<<"mode "<<mode<<", lr "<<lr<<", steps "<<steps<<": GPU "<<gpuMs<<" ms, MSE "<<double(sse)/P<<"\n";
 for(void* p:{(void*)dg,(void*)sum,(void*)moments,(void*)update,(void*)dt,(void*)quantized,(void*)audit})if(p)CK(cudaFree(p));CK(cudaEventDestroy(begin));CK(cudaEventDestroy(end));return 0;
 }catch(const std::exception& e){std::cerr<<e.what()<<"\n";return 1;}}

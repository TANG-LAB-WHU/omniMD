// Original MIT implementation. Four-component directional/parameter duals.
// CPU supplies detached dL/dE and dL/dF, normalized environment and J*cF.
struct HD { double x,u,p,m; };
__device__ HD hzero() { HD r={0.,0.,0.,0.};return r; }
__device__ HD hconstant(double x) { HD r={x,0.,0.,0.};return r; }
__device__ HD hadd(HD a,HD b) { HD r={a.x+b.x,a.u+b.u,a.p+b.p,a.m+b.m};return r; }
__device__ HD hmul(HD a,HD b) {
    HD r={a.x*b.x,a.u*b.x+a.x*b.u,a.p*b.x+a.x*b.p,
        a.m*b.x+a.u*b.p+a.p*b.u+a.x*b.m};return r;
}
__device__ HD hscale(HD a,double c) {HD r={a.x*c,a.u*c,a.p*c,a.m*c};return r;}
__device__ HD htanh(HD a) {
    double y=tanh(a.x),s=1.-y*y;
    HD r={y,s*a.u,s*a.p,s*a.m-2.*y*s*a.u*a.p};return r;
}
__device__ void hcheck(HD a,int *status) {
    if (!isfinite(a.x)||!isfinite(a.u)||!isfinite(a.p)||!isfinite(a.m)) atomicExch(status,1);
}
__device__ HD hweight(const double *w,int index,int parameter) {
    HD r={w[index],0.,index==parameter?1.:0.,0.};return r;
}
__device__ void hnet(const int *meta,const double *w,int network,int parameter,HD *x,int *status) {
    HD y[TW];int base=meta[network],layers=meta[base++];
    for (int l=0;l<layers;l++) {int k=base+l*7;
        int ni=meta[k],no=meta[k+1],wi=meta[k+2],bi=meta[k+3],ti=meta[k+4];
        for (int j=0;j<no;j++) {
            HD z=hweight(w,bi+j,parameter);
            for (int i=0;i<ni;i++) z=hadd(z,hmul(x[i],hweight(w,wi+i*no+j,parameter)));
            hcheck(z,status);
            if (meta[k+5]) z=htanh(z);
            if (ti>=0) z=hmul(z,hweight(w,ti+j,parameter));
            if (meta[k+6]) z=hadd(z,x[j%ni]);
            hcheck(z,status);y[j]=z;
        }
        for (int j=0;j<no;j++) x[j]=y[j];
    }
}
// One thread owns one (central atom, parameter); no floating atomic reduction.
extern "C" __global__ void full_gradient(const int *meta,const double *w,const double *r,
    const double *u,const int *types,const int *slot_types,int atoms,int parameters,
    double energy_seed,double *gradient,double *energy,int *status) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*parameters)return;
    int atom=q/parameters,parameter=q%parameters,kind=types[atom];
    HD b[EW*4],x[TW];for(int c=0;c<EW*4;c++)b[c]=hzero();
    for(int s=0;s<NSLOTS;s++) {
        int row=(atom*NSLOTS+s)*4;
        HD env[4];for(int a=0;a<4;a++){HD v={r[row+a],u[row+a],0.,0.};env[a]=v;}
        x[0]=env[0];hnet(meta,w,kind*NT+slot_types[s],parameter,x,status);
        for(int m=0;m<EW;m++)for(int a=0;a<4;a++)
            b[m*4+a]=hadd(b[m*4+a],hscale(hmul(x[m],env[a]),1.0/NSLOTS));
    }
    for(int c=0;c<EW*4;c++)hcheck(b[c],status);
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++) {
        HD v=hzero();for(int a=0;a<4;a++)v=hadd(v,hmul(b[m*4+a],b[k*4+a]));
        hcheck(v,status);x[m*AXIS+k]=v;
    }
    hnet(meta,w,NT*NT+kind,parameter,x,status);
    double value=energy_seed*x[0].p-x[0].m;
    if(!isfinite(value))atomicExch(status,1);
    gradient[q]=value;if(parameter==0)energy[atom]=x[0].x;
}
extern "C" __global__ void full_adam(double *w,double *first,double *second,const double *gradient,
    int parameters,double beta1,double beta2,double c1,double c2,double rate,double epsilon,int *status) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=parameters)return;
    double g=gradient[i];
    double m=beta1*first[i]+(1.-beta1)*g;
    double v=beta2*second[i]+(1.-beta2)*g*g;
    double value=w[i]-rate*(m/c1)/(sqrt(v/c2)+epsilon);
    if(!isfinite(m)||!isfinite(v)||v<0.||!isfinite(value))atomicExch(status,1);
    first[i]=m;second[i]=v;w[i]=value;
}

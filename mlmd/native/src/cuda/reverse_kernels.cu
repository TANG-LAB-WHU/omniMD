// Original MIT reverse accumulation through cached forward (value,direction) pairs.
// The scalar objective is energy_seed*E - E_direction. No floating atomics.
#define RARGS const int *meta,const double *w,const double *r,const double *u,\
 const int *types,const int *slot_types,int atoms,int parameters,double energy_seed,\
 double *gradient,double *energy,int *status,double *cache,double *badj,\
 double *slot_gradient,const int *parameter_network
__device__ void rcheck(double x,int *status) { if(!isfinite(x))atomicExch(status,1); }
__device__ int layer_count(const int *meta,int net){return meta[meta[net]];}
__device__ void reverse_forward(const int *meta,const double *w,int net,double *c,int *status) {
    int base=meta[net]+1,layers=layer_count(meta,net);
    for(int l=0;l<layers;l++) {
        int k=base+7*l,ni=meta[k],no=meta[k+1],wi=meta[k+2],bi=meta[k+3],ti=meta[k+4];
        const double *x=c+2*TW*l;double *y=c+2*TW*(l+1);
        double *activation=c+2*TW*(RL+1+l);
        for(int j=0;j<no;j++) {
            double z=w[bi+j],zu=0.;
            for(int i=0;i<ni;i++){z+=x[2*i]*w[wi+i*no+j];zu+=x[2*i+1]*w[wi+i*no+j];}
            rcheck(z,status);rcheck(zu,status);
            double h=meta[k+5]?tanh(z):z,hu=meta[k+5]?(1.-h*h)*zu:zu;
            activation[2*j]=h;activation[2*j+1]=hu;
            double dt=ti>=0?w[ti+j]:1.;
            double value=ti>=0?h*dt:h,tangent=ti>=0?hu*dt:hu;
            if(meta[k+6]){value+=x[2*(j%ni)];tangent+=x[2*(j%ni)+1];}
            rcheck(value,status);rcheck(tangent,status);y[2*j]=value;y[2*j+1]=tangent;
        }
    }
}
// Adjoint pair (dS/dx,dS/dxu), not a dual-number algebraic adjoint.
__device__ void reverse_backward(const int *meta,const double *w,int net,const double *c,
 double *adj,double *gradient,int shift,int *status) {
    int base=meta[net]+1,layers=layer_count(meta,net);double next[2*TW];
    for(int l=layers-1;l>=0;l--) {
        int k=base+7*l,ni=meta[k],no=meta[k+1],wi=meta[k+2],bi=meta[k+3],ti=meta[k+4];
        const double *x=c+2*TW*l,*activation=c+2*TW*(RL+1+l);
        for(int i=0;i<2*ni;i++)next[i]=0.;
        for(int j=0;j<no;j++) {
            double gy=adj[2*j],gu=adj[2*j+1],h=activation[2*j],hu=activation[2*j+1];
            double dt=ti>=0?w[ti+j]:1.;
            if(ti>=0){double g=gy*h+gu*hu;rcheck(g,status);gradient[ti+j-shift]=g;}
            double gh=gy*dt,ghu=gu*dt,s=meta[k+5]?1.-h*h:1.;
            double gz=meta[k+5]?gh*s-2.*h*hu*ghu:gh,gzu=ghu*s;
            rcheck(gz,status);rcheck(gzu,status);gradient[bi+j-shift]=gz;
            for(int i=0;i<ni;i++) {
                double g=gz*x[2*i]+gzu*x[2*i+1];rcheck(g,status);gradient[wi+i*no+j-shift]=g;
                next[2*i]+=w[wi+i*no+j]*gz;next[2*i+1]+=w[wi+i*no+j]*gzu;
            }
            if(meta[k+6]){next[2*(j%ni)]+=gy;next[2*(j%ni)+1]+=gu;}
        }
        for(int i=0;i<2*ni;i++){rcheck(next[i],status);adj[i]=next[i];}
    }
}
extern "C" __global__ void reverse_embed(RARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*NSLOTS)return;
    int atom=q/NSLOTS,slot=q%NSLOTS,net=types[atom]*NT+slot_types[slot];
    double *c=cache+q*RC;c[0]=r[q*4];c[1]=u[q*4];reverse_forward(meta,w,net,c,status);
}
extern "C" __global__ void reverse_fit(RARGS) {
    int atom=blockIdx.x*blockDim.x+threadIdx.x;if(atom>=atoms)return;
    int kind=types[atom],net=NT*NT+kind;
    double b[EW*8],ba[EW*8],adj[2*TW];
    for(int i=0;i<EW*8;i++){b[i]=0.;ba[i]=0.;}
    for(int s=0;s<NSLOTS;s++) {
        int q=atom*NSLOTS+s,en=kind*NT+slot_types[s];
        const double *g=cache+q*RC+2*TW*layer_count(meta,en);
        for(int m=0;m<EW;m++)for(int a=0;a<4;a++) {
            int k=2*(4*m+a);b[k]+=g[2*m]*r[q*4+a]/NSLOTS;
            b[k+1]+=(g[2*m+1]*r[q*4+a]+g[2*m]*u[q*4+a])/NSLOTS;
        }
    }
    for(int i=0;i<EW*8;i++)rcheck(b[i],status);
    double *c=cache+(atoms*NSLOTS+atom)*RC;
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++) {
        double value=0.,tangent=0.;
        for(int a=0;a<4;a++){int i=2*(4*m+a),j=2*(4*k+a);
            value+=b[i]*b[j];tangent+=b[i+1]*b[j]+b[i]*b[j+1];}
        c[2*(m*AXIS+k)]=value;c[2*(m*AXIS+k)+1]=tangent;
        rcheck(value,status);rcheck(tangent,status);
    }
    reverse_forward(meta,w,net,c,status);
    energy[atom]=c[2*TW*layer_count(meta,net)];
    for(int p=0;p<parameters;p++)gradient[atom*parameters+p]=0.;
    adj[0]=energy_seed;adj[1]=-1.;
    reverse_backward(meta,w,net,c,adj,gradient+atom*parameters,0,status);
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++) {
        double av=adj[2*(m*AXIS+k)],au=adj[2*(m*AXIS+k)+1];
        for(int a=0;a<4;a++) {int i=2*(4*m+a),j=2*(4*k+a);
            ba[i]+=av*b[j]+au*b[j+1];ba[i+1]+=au*b[j];
            ba[j]+=av*b[i]+au*b[i+1];ba[j+1]+=au*b[i];
        }
    }
    for(int i=0;i<EW*8;i++){rcheck(ba[i],status);badj[atom*EW*8+i]=ba[i];}
}
extern "C" __global__ void reverse_embedding_backward(RARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*NSLOTS)return;
    int atom=q/NSLOTS,slot=q%NSLOTS,net=types[atom]*NT+slot_types[slot];
    const double *ba=badj+atom*EW*8;double adj[2*TW];
    for(int m=0;m<EW;m++) {
        double av=0.,au=0.;
        for(int a=0;a<4;a++){int k=2*(4*m+a);
            av+=(ba[k]*r[q*4+a]+ba[k+1]*u[q*4+a])/NSLOTS;
            au+=ba[k+1]*r[q*4+a]/NSLOTS;}
        adj[2*m]=av;adj[2*m+1]=au;
    }
    int first=meta[meta[net]+3];
    reverse_backward(meta,w,net,cache+q*RC,adj,slot_gradient+q*EP,first,status);
}
extern "C" __global__ void reverse_reduce(RARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*parameters)return;
    int atom=q/parameters,p=q%parameters,net=parameter_network[p];
    if(net>=NT*NT)return; // Fitting gradients already have their unique writer.
    double sum=0.;
    if(types[atom]==net/NT) {
        int local=p-meta[meta[net]+3];
        for(int s=0;s<NSLOTS;s++)if(slot_types[s]==net%NT)sum+=slot_gradient[(atom*NSLOTS+s)*EP+local];
    }
    rcheck(sum,status);gradient[q]=sum;
}

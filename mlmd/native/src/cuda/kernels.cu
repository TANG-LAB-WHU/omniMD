// Original MIT implementation. FP64 se_e2_a network/contraction kernels.
// Model-specialized dimensions; weights and layer metadata are runtime inputs.
__device__ void bad(double x, int *status) { if (!isfinite(x)) atomicExch(status, 1); }
__device__ double activate(double z, int activation, double *slope) {
    if (activation == 1) { double y=tanh(z); *slope=1.0-y*y; return y; }
    *slope=1.0; return z;
}
extern "C" __global__ void embed(const int *meta,const double *w,const double *r,
    const int *types,const int *slot_types,int atoms,double *g,double *dg,int *status) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;
    if (q>=atoms*NSLOTS) return;
    int net=types[q/NSLOTS]*NT+slot_types[q%NSLOTS];
    int p=meta[net], layers=meta[p++];
    double x[MW],dx[MW],y[MW],dy[MW]; x[0]=r[q*4];dx[0]=1.0;
    for (int l=0;l<layers;l++,p+=7) {
        int ni=meta[p],no=meta[p+1],wi=meta[p+2],bi=meta[p+3],ti=meta[p+4];
        int act=meta[p+5],res=meta[p+6];
        for (int j=0;j<no;j++) {
            double z=w[bi+j],dz=0.0;
            for (int i=0;i<ni;i++) {z+=x[i]*w[wi+i*no+j];dz+=dx[i]*w[wi+i*no+j];}
            bad(z,status); double slope,scale=ti<0?1.0:w[ti+j];
            y[j]=activate(z,act,&slope)*scale;dy[j]=dz*slope*scale;
            if (res) {y[j]+=x[j%ni];dy[j]+=dx[j%ni];}
            bad(y[j],status);bad(dy[j],status);
        }
        for (int j=0;j<no;j++) {x[j]=y[j];dx[j]=dy[j];}
    }
    for (int m=0;m<EW;m++) {g[q*EW+m]=x[m];dg[q*EW+m]=dx[m];}
}
extern "C" __global__ void fit(const int *meta,const double *w,const double *r,
    const int *types,int atoms,const double *g,double *db_out,double *energy,int *status) {
    int atom=blockIdx.x*blockDim.x+threadIdx.x;if (atom>=atoms) return;
    double b[EW*4],db[EW*4];
    for (int i=0;i<EW*4;i++) {b[i]=0.0;db[i]=0.0;}
    for (int s=0;s<NSLOTS;s++) {int q=atom*NSLOTS+s;
        for (int m=0;m<EW;m++) for (int a=0;a<4;a++) b[m*4+a]+=g[q*EW+m]*r[q*4+a]/NSLOTS;
    }
    double x[MW],y[MW],slopes[ML*MW],adj[MW],prev[MW];
    for (int m=0;m<EW;m++) for (int k=0;k<AXIS;k++) {
        double value=0.0;for (int a=0;a<4;a++) value+=b[m*4+a]*b[k*4+a];
        x[m*AXIS+k]=value;bad(value,status);
    }
    int base=meta[NT*NT+types[atom]],layers=meta[base++];
    for (int l=0;l<layers;l++) {int p=base+l*7;
        int ni=meta[p],no=meta[p+1],wi=meta[p+2],bi=meta[p+3],ti=meta[p+4];
        int act=meta[p+5],res=meta[p+6];
        for (int j=0;j<no;j++) {
            double z=w[bi+j];for (int i=0;i<ni;i++) z+=x[i]*w[wi+i*no+j];bad(z,status);
            double slope,scale=ti<0?1.0:w[ti+j];y[j]=activate(z,act,&slope)*scale;
            slopes[l*MW+j]=slope*scale;if (res) y[j]+=x[j%ni];bad(y[j],status);
        }
        for (int j=0;j<no;j++) x[j]=y[j];
    }
    energy[atom]=x[0];adj[0]=1.0;
    for (int l=layers-1;l>=0;l--) {int p=base+l*7;
        int ni=meta[p],no=meta[p+1],wi=meta[p+2],res=meta[p+6];
        for (int i=0;i<ni;i++) {prev[i]=0.0;for (int j=0;j<no;j++) prev[i]+=adj[j]*slopes[l*MW+j]*w[wi+i*no+j];}
        if (res) for (int j=0;j<no;j++) prev[j%ni]+=adj[j];
        for (int i=0;i<ni;i++) {adj[i]=prev[i];bad(adj[i],status);}
    }
    for (int m=0;m<EW;m++) for (int k=0;k<AXIS;k++) for (int a=0;a<4;a++) {
        db[m*4+a]+=adj[m*AXIS+k]*b[k*4+a];db[k*4+a]+=adj[m*AXIS+k]*b[m*4+a];
    }
    for (int i=0;i<EW*4;i++) {db_out[atom*EW*4+i]=db[i];bad(db[i],status);}
}
extern "C" __global__ void pullback(const double *r,int atoms,const double *g,
    const double *dg,const double *db,double *dr,int *status) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if (q>=atoms*NSLOTS) return;
    int atom=q/NSLOTS;double out[4]={0.0,0.0,0.0,0.0};
    for (int m=0;m<EW;m++) for (int a=0;a<4;a++) {
        out[a]+=db[atom*EW*4+m*4+a]*g[q*EW+m]/NSLOTS;
        out[0]+=db[atom*EW*4+m*4+a]*r[q*4+a]*dg[q*EW+m]/NSLOTS;
    }
    for (int a=0;a<4;a++) {dr[q*4+a]=out[a];bad(out[a],status);}
}

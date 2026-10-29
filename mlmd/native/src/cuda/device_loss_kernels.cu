// Original MIT per-tile prediction and detached loss seeds.
// Primal values/activations survive until the mixed reverse has completed.
#define LARGS const int *meta,const double *w,const double *r,double *u, \
 const int *types,const int *slot_types,int atoms,int parameters,double *energy_seed, \
 double *gradient,double *energy,int *status,double *cache,double *badj, \
 double *slot_gradient,const int *parameter_network,const int *owners, \
 const double *geometry,const int *frame_ids,const int *bounds, \
 const double *force_labels,const double *frame_labels,const double *scales, \
 double *force_pred,double *frame_pred,double *force_seed,double *frame_seed, \
 double *edge_grad,double *bvalue,const double *offsets,int frames

__device__ void loss_input_backward(const int *meta,const double *w,int net,
 const double *c,double *adj,int *status) {
    int base=meta[net]+1,layers=layer_count(meta,net);double next[TW];
    for(int l=layers-1;l>=0;l--) {
        int k=base+7*l,ni=meta[k],no=meta[k+1],wi=meta[k+2],ti=meta[k+4];
        const double *activation=c+2*TW*(RL+1+l);
        for(int i=0;i<ni;i++)next[i]=0.;
        for(int j=0;j<no;j++) {
            double h=activation[2*j],gy=adj[j];
            double gh=gy*(ti>=0?w[ti+j]:1.);
            double gz=meta[k+5]?gh*(1.-h*h):gh;rcheck(gz,status);
            for(int i=0;i<ni;i++)next[i]+=w[wi+i*no+j]*gz;
            if(meta[k+6])next[j%ni]+=gy;
        }
        for(int i=0;i<ni;i++){rcheck(next[i],status);adj[i]=next[i];}
    }
}
__device__ void loss_forward_direction(const int *meta,const double *w,int net,
 double *c,int *status) {
    int base=meta[net]+1,layers=layer_count(meta,net);
    for(int l=0;l<layers;l++) {
        int k=base+7*l,ni=meta[k],no=meta[k+1],wi=meta[k+2],ti=meta[k+4];
        const double *x=c+2*TW*l;double *y=c+2*TW*(l+1);
        double *activation=c+2*TW*(RL+1+l);
        for(int j=0;j<no;j++) {
            double zu=0.;for(int i=0;i<ni;i++)zu+=x[2*i+1]*w[wi+i*no+j];
            double h=activation[2*j],hu=meta[k+5]?(1.-h*h)*zu:zu;
            double tangent=ti>=0?hu*w[ti+j]:hu;
            if(meta[k+6])tangent+=x[2*(j%ni)+1];
            rcheck(zu,status);rcheck(hu,status);rcheck(tangent,status);
            activation[2*j+1]=hu;y[2*j+1]=tangent;
        }
    }
}
extern "C" __global__ void loss_primal_embed(LARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*NSLOTS)return;
    int atom=q/NSLOTS,net=types[atom]*NT+slot_types[q%NSLOTS];
    double *c=cache+q*RC;c[0]=r[q*4];c[1]=0.;
    reverse_forward(meta,w,net,c,status);
}
extern "C" __global__ void loss_primal_fit(LARGS) {
    int atom=blockIdx.x*blockDim.x+threadIdx.x;if(atom>=atoms)return;
    int kind=types[atom],net=NT*NT+kind;
    double b[EW*4],ba[EW*4],adj[TW];
    for(int i=0;i<EW*4;i++){b[i]=0.;ba[i]=0.;}
    for(int s=0;s<NSLOTS;s++) {
        int q=atom*NSLOTS+s,en=kind*NT+slot_types[s];
        const double *g=cache+q*RC+2*TW*layer_count(meta,en);
        for(int m=0;m<EW;m++)for(int a=0;a<4;a++)b[4*m+a]+=g[2*m]*r[q*4+a]/NSLOTS;
    }
    for(int i=0;i<EW*4;i++){rcheck(b[i],status);bvalue[atom*EW*4+i]=b[i];}
    double *c=cache+(atoms*NSLOTS+atom)*RC;
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++) {
        double value=0.;for(int a=0;a<4;a++)value+=b[4*m+a]*b[4*k+a];
        rcheck(value,status);c[2*(m*AXIS+k)]=value;c[2*(m*AXIS+k)+1]=0.;
    }
    reverse_forward(meta,w,net,c,status);
    energy[atom]=c[2*TW*layer_count(meta,net)];
    adj[0]=1.;loss_input_backward(meta,w,net,c,adj,status);
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++)for(int a=0;a<4;a++){
        ba[4*m+a]+=adj[m*AXIS+k]*b[4*k+a];
        ba[4*k+a]+=adj[m*AXIS+k]*b[4*m+a];
    }
    for(int i=0;i<EW*4;i++){rcheck(ba[i],status);badj[atom*EW*8+2*i]=ba[i];}
}
extern "C" __global__ void loss_edge_gradient(LARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*NSLOTS)return;
    if(owners[q]<0){for(int b=0;b<3;b++)edge_grad[q*3+b]=0.;return;}
    int atom=q/NSLOTS,net=types[atom]*NT+slot_types[q%NSLOTS];
    const double *ba=badj+atom*EW*8,*g=cache+q*RC+2*TW*layer_count(meta,net);
    double dr[4]={0.,0.,0.,0.},adj[TW];
    for(int m=0;m<EW;m++){
        double a=0.;
        for(int k=0;k<4;k++){dr[k]+=ba[2*(4*m+k)]*g[2*m]/NSLOTS;a+=ba[2*(4*m+k)]*r[q*4+k]/NSLOTS;}
        adj[m]=a;
    }
    loss_input_backward(meta,w,net,cache+q*RC,adj,status);
    dr[0]+=adj[0]; // embedding-input path, in addition to direct R dependence
    for(int b=0;b<3;b++){
        double v=0.;for(int a=0;a<4;a++)v+=dr[a]*geometry[q*15+3*a+b];
        rcheck(v,status);edge_grad[q*3+b]=v;
    }
}
extern "C" __global__ void loss_predictions(LARGS) {
    int f=blockIdx.x,t=threadIdx.x;if(f>=frames)return;
    int begin=bounds[2*f],end=bounds[2*f+1],n=end-begin;
    if(t==0){
        double e=0.;for(int i=begin;i<end;i++)e+=energy[i]+offsets[types[i]];
        rcheck(e,status);frame_pred[f*10]=e;
    }else if(t<=3*n){
        int atom=begin+(t-1)/3,b=(t-1)%3;double force=0.;
        for(int q=begin*NSLOTS;q<end*NSLOTS;q++){
            if(owners[q]<0)continue;
            double g=edge_grad[q*3+b];
            if(q/NSLOTS==atom)force+=g;
            if(owners[q]==atom)force-=g;
        }
        rcheck(force,status);force_pred[atom*3+b]=force;
    }else if(t<=3*n+9){
        int k=t-3*n-1,a=k/3,b=k%3;double v=0.;
        for(int q=begin*NSLOTS;q<end*NSLOTS;q++)if(owners[q]>=0)v-=geometry[q*15+12+a]*edge_grad[q*3+b];
        rcheck(v,status);frame_pred[f*10+1+k]=v;
    }
}
extern "C" __global__ void loss_seeds(LARGS) {
    int f=blockIdx.x*blockDim.x+threadIdx.x;if(f>=frames)return;
    int begin=bounds[2*f],end=bounds[2*f+1];
    double es=scales[3*f],fs=scales[3*f+1],vs=scales[3*f+2];
    double de=frame_pred[f*10]-frame_labels[f*10],a=2.*es*es*de;
    double z=de*es,objective=z*z;rcheck(a,status);
    frame_seed[f*11]=a;
    for(int i=begin;i<end;i++){
        energy_seed[i]=a;
        for(int b=0;b<3;b++){
            double df=force_pred[i*3+b]-force_labels[i*3+b];
            double seed=2.*fs*fs*df;z=df*fs;objective+=z*z;
            rcheck(seed,status);force_seed[i*3+b]=seed;
        }
    }
    for(int k=0;k<9;k++){
        double seed=0.;
        if(vs>0.){
            double dv=frame_pred[f*10+1+k]-frame_labels[f*10+1+k];
            seed=2.*vs*vs*dv;z=dv*vs;objective+=z*z;
        }
        rcheck(seed,status);frame_seed[f*11+1+k]=seed;
    }
    rcheck(objective,status);frame_seed[f*11+10]=objective;
}
extern "C" __global__ void loss_directions(LARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*NSLOTS)return;
    int atom=q/NSLOTS,j=owners[q],f=frame_ids[atom];
    for(int a=0;a<4;a++){
        double tangent=0.;
        if(j>=0)for(int b=0;b<3;b++){
            double direction=force_seed[j*3+b]-force_seed[atom*3+b];
            if(scales[3*f+2]>0.){
                double strain=0.;for(int k=0;k<3;k++)strain+=geometry[q*15+12+k]*frame_seed[f*11+1+3*k+b];
                direction+=strain;
            }
            tangent+=geometry[q*15+3*a+b]*direction;
        }
        rcheck(tangent,status);u[q*4+a]=tangent;
    }
}
extern "C" __global__ void loss_dual_embed(LARGS) {
    int q=blockIdx.x*blockDim.x+threadIdx.x;if(q>=atoms*NSLOTS)return;
    int atom=q/NSLOTS,net=types[atom]*NT+slot_types[q%NSLOTS];
    double *c=cache+q*RC;c[1]=u[q*4];
    loss_forward_direction(meta,w,net,c,status);
}
extern "C" __global__ void loss_dual_fit(LARGS) {
    int atom=blockIdx.x*blockDim.x+threadIdx.x;if(atom>=atoms)return;
    int kind=types[atom],net=NT*NT+kind;
    double b[EW*8],ba[EW*8],adj[2*TW];
    for(int i=0;i<EW*4;i++){b[2*i]=bvalue[atom*EW*4+i];b[2*i+1]=0.;ba[2*i]=0.;ba[2*i+1]=0.;}
    for(int s=0;s<NSLOTS;s++){
        int q=atom*NSLOTS+s,en=kind*NT+slot_types[s];
        const double *g=cache+q*RC+2*TW*layer_count(meta,en);
        for(int m=0;m<EW;m++)for(int a=0;a<4;a++){
            int k=2*(4*m+a);b[k+1]+=(g[2*m+1]*r[q*4+a]+g[2*m]*u[q*4+a])/NSLOTS;
        }
    }
    for(int i=0;i<EW*8;i++)rcheck(b[i],status);
    double *c=cache+(atoms*NSLOTS+atom)*RC;
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++){
        double tangent=0.;for(int a=0;a<4;a++){int i=2*(4*m+a),j=2*(4*k+a);tangent+=b[i+1]*b[j]+b[i]*b[j+1];}
        rcheck(tangent,status);c[2*(m*AXIS+k)+1]=tangent;
    }
    loss_forward_direction(meta,w,net,c,status);
    for(int p=0;p<parameters;p++)gradient[atom*parameters+p]=0.;
    adj[0]=energy_seed[atom];adj[1]=-1.;
    reverse_backward(meta,w,net,c,adj,gradient+atom*parameters,0,status);
    for(int m=0;m<EW;m++)for(int k=0;k<AXIS;k++){
        double av=adj[2*(m*AXIS+k)],au=adj[2*(m*AXIS+k)+1];
        for(int a=0;a<4;a++){int i=2*(4*m+a),j=2*(4*k+a);
            ba[i]+=av*b[j]+au*b[j+1];ba[i+1]+=au*b[j];
            ba[j]+=av*b[i]+au*b[i+1];ba[j+1]+=au*b[i];}
    }
    for(int i=0;i<EW*8;i++){rcheck(ba[i],status);badj[atom*EW*8+i]=ba[i];}
}

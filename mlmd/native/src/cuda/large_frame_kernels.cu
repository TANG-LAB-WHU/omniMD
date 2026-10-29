// Original MIT large-frame gather. No atom-sized local array; each output keeps its prior sum order.
extern "C" __global__ void loss_predictions_large(LARGS) {
    int f=blockIdx.x;if(f>=frames)return;
    int begin=bounds[2*f],end=bounds[2*f+1],n=end-begin;
    for(int t=threadIdx.x;t<1+3*n+9;t+=blockDim.x) {
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
}

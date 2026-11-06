"""Differentiable geometric feature adapter; NOT an analytic root update.

From existing channel-first 5x5 residual block P=(dx,dy,d)/2, let r
be its central directional residual, a and b its centered finite differences
in u and -v. With h=1/47, s=sqrt(||a||^2+||b||^2+epsilon), append
H(P)=[r/(hs),a/s,b/s,<a,b>/(||a||||b||),det(a,b)/(||a||||b||)] in R^8.
No matrix inverse, linear-system solve, root correction formula or geometry
threshold appears here. The MLP still predicts every displacement.
The adapter backward includes the normalization derivative exactly.
"""
import numpy as np
from code.models.mlp_numpy import make_mlp

H=1/47
EPS=1e-10


def encode(p):
    r=p[:,[12,37]]
    a=(p[:,[17,42]]-p[:,[7,32]])/(2*H)
    b=-(p[:,[13,38]]-p[:,[11,36]])/(2*H)
    s=np.sqrt(np.sum(a*a+b*b,axis=1,keepdims=True)+EPS)
    na=np.sqrt(np.sum(a*a,axis=1,keepdims=True)+EPS)
    nb=np.sqrt(np.sum(b*b,axis=1,keepdims=True)+EPS)
    dot=np.sum(a*b,axis=1,keepdims=True)
    cross=(a[:,0]*b[:,1]-a[:,1]*b[:,0])[:,None]
    den=na*nb
    out=np.c_[r/(H*s),a/s,b/s,dot/den,cross/den]
    return out,(r,a,b,s,na,nb,dot,cross,den)


def backward(gradient,cache):
    r,a,b,s,na,nb,dot,cross,den=cache
    gr=gradient[:,:2]/(H*s)
    ga=gradient[:,2:4]/s
    gb=gradient[:,4:6]/s
    gs=-np.sum(gradient[:,:2]*r/H+gradient[:,2:4]*a+gradient[:,4:6]*b,axis=1,keepdims=True)/(s*s)
    ga+=gs*a/s;gb+=gs*b/s
    gc,gd=gradient[:,6:7],gradient[:,7:8]
    ga+=gc*b/den+gd*np.c_[b[:,1],-b[:,0]]/den
    gb+=gc*a/den+gd*np.c_[-a[:,1],a[:,0]]/den
    gden=-(gc*dot+gd*cross)/(den*den)
    ga+=gden*nb*a/na;gb+=gden*na*b/nb
    gp=np.zeros((len(r),75))
    gp[:,[12,37]]+=gr
    gp[:,[17,42]]+=ga/(2*H);gp[:,[7,32]]-=ga/(2*H)
    gp[:,[13,38]]-=gb/(2*H);gp[:,[11,36]]+=gb/(2*H)
    return gp


class GeometryFeatureNetwork:
    """Expose77-dimensional interface to BPTT; internal MLP has85 inputs."""
    def __init__(self,seed):
        self.network=make_mlp(85,[128,128,64],3,np.random.default_rng(seed))
        self.cache=[]

    def forward(self,inputs):
        extra,cache=encode(inputs[:,2:]);self.cache.append(cache)
        return self.network.forward(np.c_[inputs,extra])

    def backward(self,gradient):
        expanded=self.network.backward(gradient)
        result=expanded[:,:77].copy()
        result[:,2:]+=backward(expanded[:,77:],self.cache.pop())
        return result

    def clear_cache(self):
        self.cache.clear();self.network.clear_cache()

    def zero_grad(self):self.network.zero_grad()
    def step(self,*args):self.network.step(*args)
    def parameter_layers(self):return self.network.parameter_layers()
    def state(self,prefix):return self.network.state(prefix)

    def load_state(self,state,prefix):
        state={k:np.array(v,copy=True) for k,v in state.items()}
        key=prefix+'.dense0.weight'
        if state[key].shape[0]==77:
            state[key]=np.pad(state[key],((0,8),(0,0)))
        self.network.load_state(state,prefix)


def check():
    from code.experiments import learned_correction as lc
    rng=np.random.default_rng(20261026)
    p=rng.normal(0,.03,(3,75));g=rng.normal(size=(3,8))
    out,cache=encode(p);analytic=backward(g,cache);error=0
    for col in [7,11,12,13,17,32,36,37,38,42]:
        plus=p.copy();minus=p.copy();plus[0,col]+=1e-6;minus[0,col]-=1e-6
        numeric=float(np.sum((encode(plus)[0]-encode(minus)[0])*g)/2e-6)
        error=max(error,abs(numeric-analytic[0,col]))
    assert error<2e-6,error
    original=lc.LearnedCorrectionModel(seed=3,step_scale=.02,dynamic_patches=True)
    adapted=lc.LearnedCorrectionModel(seed=3,step_scale=.02,dynamic_patches=True)
    adapted.network=GeometryFeatureNetwork(3);adapted.load_state(original.state())
    x=rng.normal(0,.03,(4,77));x[:,:2]=rng.uniform(.3,.7,(4,2))
    a,b=rng.normal(0,.3,(2,4,8,2))
    target=np.c_[rng.uniform(.3,.7,(4,2)),[1,0,1,1]]
    before=lc.unroll(original,x,4,curve_a=a,curve_b=b)[0]
    after=lc.unroll(adapted,x,4,curve_a=a,curve_b=b)[0]
    parity=float(np.max(np.abs(before-after)));assert parity<1e-14,parity
    args=(x,target,a,b,4,1,1000,5,0)
    lc.batch_loss_and_gradient(adapted,*args)
    layer=adapted.network.parameter_layers()[0]
    row,col=80,0
    # Pick a nonzero new-feature weight gradient to exercise the added path.
    row,col=np.unravel_index(np.argmax(np.abs(layer.grad_weight[77:])),layer.grad_weight[77:].shape);row+=77
    analytic=float(layer.grad_weight[row,col]);initial=layer.weight[row,col]
    layer.weight[row,col]=initial+1e-6;plus=lc.batch_loss_and_gradient(adapted,*args)[0]
    layer.weight[row,col]=initial-1e-6;minus=lc.batch_loss_and_gradient(adapted,*args)[0]
    layer.weight[row,col]=initial
    first_error=abs((plus-minus)/2e-6-analytic);assert first_error<2e-5,first_error
    # Nonzero adapter weights: verify full quotient-chain BPTT as well.
    layer.weight[77:]=rng.normal(0,.02,layer.weight[77:].shape)
    lc.batch_loss_and_gradient(adapted,*args)
    row,col=np.unravel_index(np.argmax(np.abs(layer.grad_weight)),layer.grad_weight.shape)
    analytic=float(layer.grad_weight[row,col]);initial=layer.weight[row,col]
    layer.weight[row,col]=initial+1e-6;plus=lc.batch_loss_and_gradient(adapted,*args)[0]
    layer.weight[row,col]=initial-1e-6;minus=lc.batch_loss_and_gradient(adapted,*args)[0]
    full_error=abs((plus-minus)/2e-6-analytic);assert full_error<2e-5,full_error
    print(dict(feature_gradient_error=error,initial_trajectory_difference=parity,
               new_feature_weight_gradient_error=first_error,full_bptt_gradient_error=full_error))


if __name__=='__main__':check()

"""Curve-only API; preserve the historical learned four-step solver exactly.

External input: two (8,2) control-point arrays (degree/knots/weights fixed).
Internal feature x_t = [q_t, vec(R(q_t))] in R^77, channel-first R/2.
q_(t+1) = clip(q_t + h*tanh(f_W(x_t)[:2]), 0, 1).
External output: final absolute q_T and sigmoid(final existence logit).
The deterministic grid layer is inside this API, but is not learnable.
"""
import numpy as np
from code.experiments import learned_correction as lc
from code.experiments import candidate_model as cm
from code.geometry.vectorized_basis import basis_matrix
from code.models.mlp_numpy import sigmoid


def features_for_pair(a, b, config):
    if config['grid_kind'] == 'uniform' and not config['boundary_candidates']:
        return cm.candidate_features(a, b, 48, 0.015, 8)
    n = 48
    if config['grid_kind'] == 'knot_aligned':
        # Five equal knot spans, 48 points total, with knot endpoints exact.
        values = np.r_[np.linspace(0, .2, 10, endpoint=False),
                       np.linspace(.2, .4, 10, endpoint=False),
                       np.linspace(.4, .6, 9, endpoint=False),
                       np.linspace(.6, .8, 9, endpoint=False), np.linspace(.8, 1, 10)]
    elif config['grid_kind'] == 'endpoint_dense':
        values = .5 * (1 - np.cos(np.linspace(0, np.pi, n)))
    else:
        values = np.linspace(0, 1, n)
    pa, pb = basis_matrix(values) @ a, basis_matrix(values) @ b
    residual = pa[:, None] - pb[None, :]
    distance = np.linalg.norm(residual, axis=-1)
    if config['boundary_candidates']:
        padded = np.pad(distance, 1, constant_values=np.inf)
        minimum = np.minimum.reduce([padded[i:i+n, j:j+n] for i in range(3) for j in range(3)])
        indices = np.argwhere(distance <= minimum + .015)
        ranked = sorted((float(distance[i,j]),int(i),int(j)) for i,j in indices)
        chosen = []
        for item in ranked:
            _, i, j = item
            if all(abs(values[i]-values[k]) > .04 or abs(values[j]-values[l]) > .04 for _,k,l in chosen):
                chosen.append(item)
            if len(chosen) == 8:
                break
        while len(chosen) < 8:
            chosen.append((float(distance.max()),n//2,n//2))
    else:
        chosen = cm.candidate_indices(distance, 8, .015)
    centers = np.array([[values[i],values[j]] for _,i,j in chosen])
    # Dynamic-patch solver recomputes this at t=0; preserve its offset units.
    patch, _, _ = lc.dynamic_patch(centers, np.repeat(a[None],8,axis=0), np.repeat(b[None],8,axis=0))
    return np.c_[centers,patch], centers


class IntegratedIntersectionMLP:
    def __init__(self, config, state):
        self.config = config
        self.solver = lc.LearnedCorrectionModel(seed=config['seed'], step_scale=config['step_scale'], dynamic_patches=True)
        if config.get('geometry_features',False):
            from code.experiments.autoresearch_nurbs.geometry_features import GeometryFeatureNetwork
            self.solver.network=GeometryFeatureNetwork(config['seed'],config.get('residual_projections',False))
        self.solver.load_state(state)

    def __call__(self, curve_a, curve_b):
        a, b = np.asarray(curve_a), np.asarray(curve_b)
        if a.shape != (8,2) or b.shape != (8,2):
            raise ValueError('This experiment requires two cubic 8-control-point curves.')
        features, centers = features_for_pair(a,b,self.config)
        parameters, records, _ = lc.unroll(self.solver,features,self.config['steps'],
            curve_a=np.repeat(a[None],len(features),axis=0),curve_b=np.repeat(b[None],len(features),axis=0))
        scores = sigmoid(records[-1][0][:,2])
        self.solver.network.clear_cache()
        return parameters, scores, centers


def prepare(curves_a,curves_b,labels,counts,config):
    rows = {k:[] for k in ('features','target','curve_a','curve_b','groups')}
    for i,(a,b,label,count) in enumerate(zip(curves_a,curves_b,labels,counts)):
        features, centers = features_for_pair(a,b,config)
        target = np.zeros((len(centers),3))
        if count:
            distances = np.linalg.norm(centers[:,None]-label[None,:int(count)],axis=-1)
            nearest = distances.argmin(axis=1)
            positive = distances[np.arange(len(centers)),nearest] <= .085
            target[positive,:2] = label[nearest[positive]]
            target[positive,2] = 1
        rows['features'].append(features)
        rows['target'].append(target)
        rows['curve_a'].append(np.repeat(a[None],len(centers),axis=0))
        rows['curve_b'].append(np.repeat(b[None],len(centers),axis=0))
        rows['groups'].append(np.full(len(centers),i))
    return {k:np.concatenate(v) for k,v in rows.items()}

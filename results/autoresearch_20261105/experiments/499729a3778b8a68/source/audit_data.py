"""Independent offline train/validation audit; never used by inference.

Numerical reference labelling and constrained distance minimization are audit
tools, not part of the neural solver. A local minimum is an upper bound on the
true global closest distance, and agreement between two sampling densities is
evidence of label stability, not a mathematical completeness certificate.
"""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from code.geometry.vectorized_basis import basis_matrix,basis_derivative_matrix
from code.data.locked_reference import intersections
from code.evaluation.set_metrics import matching

ROOT=Path(__file__).resolve().parents[3]
TOLS=(.035,.01)


def canonical_key(a,b):
    keys=[]
    for first,second in ((a,b),(b,a)):
        points=np.r_[first,second];points=points-points.mean(axis=0)
        direction=first[-1]-first[0]
        if np.linalg.norm(direction)<1e-10:direction=first[1]-first[0]
        direction=direction/max(np.linalg.norm(direction),1e-12)
        frame=np.array([[direction[0],direction[1]],[-direction[1],direction[0]]])
        normalized=points@frame.T/max(np.linalg.norm(points),1e-12)
        keys.append(hashlib.sha256(np.round(normalized,8).tobytes()).hexdigest())
    return min(keys)


def closest_distance_upper_bound(a,b):
    values=np.linspace(0,1,49)
    pa,pb=basis_matrix(values)@a,basis_matrix(values)@b
    distance=np.linalg.norm(pa[:,None]-pb[None,:],axis=-1)
    candidates=[]
    for flat in np.argsort(distance,axis=None):
        i,j=np.unravel_index(flat,distance.shape);q=np.array([values[i],values[j]])
        if all(np.linalg.norm(q-p)>.15 for p in candidates):candidates.append(q)
        if len(candidates)==6:break
    def objective(q):
        pointa=basis_matrix([q[0]])@a;pointb=basis_matrix([q[1]])@b
        da=basis_derivative_matrix([q[0]])@a;db=basis_derivative_matrix([q[1]])@b
        r=(pointa-pointb)[0]
        return float(r@r),np.array([2*r@da[0],-2*r@db[0]])
    best=float(distance.min());bestq=candidates[0].copy();success=0
    for q in candidates:
        result=minimize(objective,q,jac=True,method='L-BFGS-B',bounds=((0,1),(0,1)),
                        options={'ftol':1e-15,'gtol':1e-10,'maxiter':100})
        success+=int(result.success)
        value=float(np.sqrt(max(objective(result.x)[0],0)))
        if value<best:best=value;bestq=result.x.copy()
    return dict(distance_upper_bound=best,parameters=bestq.tolist(),starts=len(candidates),
                converged_starts=success,global_minimum_certified=False)


def run(output):
    archive=np.load(ROOT/'data/experiment_large.npz');seen={};duplicates=[];result={}
    for split in ('train','validation'):
        rows=[];categories=Counter();unstable=[];residuals=[]
        for i,(a,b,label,count) in enumerate(zip(*(archive[split+'_'+name] for name in ('curve_a','curve_b','labels','counts')))):
            key=canonical_key(a,b)
            if key in seen and seen[key][0]!=split:duplicates.append([seen[key],[split,i]])
            seen[key]=[split,i]
            truth=label[:int(count)]
            independent=intersections(a,b,241)
            stable=len(independent)==int(count) and len(matching(independent,truth,1e-5))==int(count)
            if not stable:unstable.append(i)
            row=dict(index=i,root_count=int(count),reference_count_241=len(independent),
                     sampling_stable=stable,reference='offline polyline241 plus numerical refinement')
            if count:
                ra=basis_matrix(truth[:,0])@a-basis_matrix(truth[:,1])@b
                residual=np.linalg.norm(ra,axis=1);residuals.extend(residual.tolist())
                endpoint=np.min(np.c_[truth,1-truth],axis=1)<=1e-6
                da=basis_derivative_matrix(truth[:,0])@a;db=basis_derivative_matrix(truth[:,1])@b
                angle=np.abs(da[:,0]*db[:,1]-da[:,1]*db[:,0])/np.maximum(np.linalg.norm(da,axis=1)*np.linalg.norm(db,axis=1),1e-12)
                tags=['endpoint' if edge else ('near_tangent' if value<=1e-3 else 'interior_crossing') for edge,value in zip(endpoint,angle)]
                categories.update(tags)
                row.update(root_types=tags,root_residuals=residual.tolist(),label_pass={str(t):bool(np.all(residual<=t)) for t in TOLS})
                row['closest_distance_note']='A verified root witnesses minimum distance <= its residual.'
            else:
                minimum=closest_distance_upper_bound(a,b);row['closest_distance_estimate']=minimum
                d=minimum['distance_upper_bound']
                tag='near_disjoint_within_0p01' if d<=.01 else ('near_disjoint_within_0p035' if d<=.035 else 'disjoint_far_estimate')
                categories[tag]+=1;row['distance_category']=tag
                row['topological_label']='disjoint; proximity tolerance does not relabel it as intersection'
            rows.append(row)
            if i%200==0:print(f'audit {split} {i}/{len(archive[split+"_counts"])}',flush=True)
        result[split]=dict(pairs=len(rows),categories=dict(categories),unstable_indices=unstable,
                           root_residual_max=max(residuals,default=0),
                           root_label_pass_rate={str(t):float(np.mean(np.asarray(residuals)<=t)) for t in TOLS},rows=rows)
    result['cross_split_translation_rotation_scale_and_swap_fingerprint_duplicates']=duplicates
    result['no_test_arrays_indexed']=True
    result['label_or_split_changed']=False
    result['limitations']='Numerical sampling stability and multistart distance upper bounds; no certified root completeness or certified global distance.'
    result['data_sha256']=hashlib.sha256((ROOT/'data/experiment_large.npz').read_bytes()).hexdigest()
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({k:{x:v[x] for x in ('pairs','categories','unstable_indices','root_residual_max')} for k,v in result.items() if k in ('train','validation')}),flush=True)
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--output',default=str(ROOT/'.tmp/autoresearch_nurbs/data_audit.json'))
    run(parser.parse_args().output)

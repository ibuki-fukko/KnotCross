"""Frozen validation-only protocol. No test arrays are indexed by this module.

Metric: max validation F1@.01 over fixed thresholds and saved epochs,
subject to validation F1@.035 >= .88. Both measured on the same predictions.
Special-family validation is reported separately, never called a final test.
"""
import os
# Set before numpy imports. One thread makes small-matrix training repeatable/fast.
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import argparse
import hashlib
import json
import platform
from pathlib import Path
import time
import numpy as np
from code.evaluation.set_metrics import match_metrics, matching
from code.experiments import learned_correction as lc
from code.experiments import candidate_model as cm
from code.experiments.autoresearch_nurbs.model import IntegratedIntersectionMLP, prepare
from code.experiments.autoresearch_nurbs.runner import train_or_load, ROOT

THRESHOLDS = np.round(np.arange(.05,.991,.01),2)
SUPPRESSION = .025
TOLERANCES = (.035,.01)


def load_validation():
    archive=np.load(ROOT/'data/experiment_large.npz')
    return {k:archive['validation_'+k] for k in ('curve_a','curve_b','labels','counts')}


def outputs(model,data):
    rows=[]
    for a,b in zip(data['curve_a'],data['curve_b']):
        rows.append(model(a,b))
    return rows


def decode(rows,threshold):
    predicted=[]
    for parameters,scores,*_ in rows:
        points=[]
        for i in np.argsort(-scores):
            if scores[i] < threshold: break
            if all(np.linalg.norm(parameters[i]-p)>SUPPRESSION for p in points):
                points.append(parameters[i].copy())
        predicted.append(points)
    return predicted


def pair_metrics(pred,data):
    return {str(t):match_metrics(pred,data['labels'],data['counts'],t) for t in TOLERANCES}


def select(rows,data):
    choices=[]
    for threshold in THRESHOLDS:
        pred=decode(rows,float(threshold)); metrics=pair_metrics(pred,data)
        if metrics['0.035']['f1']>=.88:
            choices.append((metrics['0.01']['f1'],metrics['0.035']['f1'],-float(threshold),metrics))
    if not choices:
        return None
    score,_,negative_threshold,metrics=max(choices,key=lambda x:x[:3])
    return dict(score=score,threshold=-negative_threshold,metrics=metrics)


def quality(data):
    residuals=[]; endpoints=0
    for a,b,label,count in zip(data['curve_a'],data['curve_b'],data['labels'],data['counts']):
        roots=label[:int(count)]
        if not len(roots):continue
        r=lc.nurbs_points_batch(np.repeat(a[None],len(roots),axis=0),roots[:,0])-lc.nurbs_points_batch(np.repeat(b[None],len(roots),axis=0),roots[:,1])
        residuals.extend(np.linalg.norm(r,axis=1).tolist())
        endpoints+=int(np.sum(np.min(np.c_[roots,1-roots],axis=1)<=1e-6))
    return dict(pairs=len(data['counts']),roots=int(sum(data['counts'])),endpoint_roots=endpoints,
                label_residual_max=max(residuals,default=0),
                label_geometric_pass={str(t):float(np.mean(np.array(residuals)<=t)) for t in TOLERANCES},
                note='Residual agreement checks given labels; does not certify completeness of legacy numerical labels.')


def evaluate():
    config,key,out=train_or_load()
    if (out/'evaluation.json').exists():
        return json.loads((out/'evaluation.json').read_text(encoding='utf-8'))
    data=load_validation(); candidates=[]; epoch_rows=[]
    for checkpoint in sorted(out.glob('epoch_*.npz')):
        model=IntegratedIntersectionMLP(config,dict(np.load(checkpoint)))
        started=time.perf_counter(); rows=outputs(model,data); seconds=time.perf_counter()-started
        choice=select(rows,data)
        epoch_rows.append(dict(checkpoint=checkpoint.name,selection=choice))
        if choice:
            candidates.append((choice['score'],choice['metrics']['0.035']['f1'],checkpoint,choice,rows,seconds))
    if not candidates:
        result=dict(key=key,score=0.0,guard_pass=False,epoch_selections=epoch_rows)
    else:
        _,_,checkpoint,choice,rows,seconds=max(candidates,key=lambda x:(x[0],x[1]))
        model=IntegratedIntersectionMLP(config,dict(np.load(checkpoint)))
        np.savez_compressed(out/'weights.npz',**model.solver.state())
        pred=decode(rows,choice['threshold'])
        featuredata=prepare(data['curve_a'],data['curve_b'],data['labels'],data['counts'],config)
        q,rec,_=lc.unroll(model.solver,featuredata['features'],config['steps'],curve_a=featuredata['curve_a'],curve_b=featuredata['curve_b'])
        labels=featuredata['target'][:,2]
        probabilities=np.clip(1/(1+np.exp(-rec[-1][0][:,2])),1e-7,1-1e-7)
        bce=float(-np.mean((1+6*labels)*(labels*np.log(probabilities)+(1-labels)*np.log(1-probabilities))))
        reg,geo,_=lc.state_loss_and_gradient(q,featuredata['target'],featuredata['curve_a'],featuredata['curve_b'],1,4)
        model.solver.network.clear_cache()
        special_archive=np.load(ROOT/'data/meeting_analytic_validation.npz')
        special={k:special_archive[k] for k in ('curve_a','curve_b','labels','counts')}
        metadata=json.loads((ROOT/'data/meeting_analytic_validation.json').read_text(encoding='utf-8'))['metadata']
        special_pred=decode(outputs(model,special),choice['threshold']); by_category={}
        for category in sorted(set(r['category'] for r in metadata)):
            ids=[i for i,r in enumerate(metadata) if r['category']==category]
            sub={k:v[ids] for k,v in special.items()}
            metrics=pair_metrics([special_pred[i] for i in ids],sub)
            if not np.sum(sub['counts']):
                metrics['empty_false_positive_rate']=float(np.mean([bool(special_pred[i]) for i in ids]))
            by_category[category]=metrics
        residuals=[]
        coverage={str(t):0 for t in TOLERANCES}
        for row,a,b,label,count,points in zip(rows,data['curve_a'],data['curve_b'],data['labels'],data['counts'],pred):
            for t in TOLERANCES:
                coverage[str(t)]+=len(matching(row[2],label[:int(count)],t))
            if points:
                points=np.asarray(points)
                delta=lc.nurbs_points_batch(np.repeat(a[None],len(points),axis=0),points[:,0])-lc.nurbs_points_batch(np.repeat(b[None],len(points),axis=0),points[:,1])
                residuals.extend(np.linalg.norm(delta,axis=1).tolist())
        result=dict(key=key,score=choice['score'],guard_pass=True,config=config,checkpoint=checkpoint.name,
                    threshold=choice['threshold'],metrics=choice['metrics'],
                    validation_reference_loss=bce+reg+geo,loss_components=dict(bce_weight7=bce,parameter_weight4=reg,geometry_weight1=geo),
                    candidate_one_to_one_coverage={k:v/int(sum(data['counts'])) for k,v in coverage.items()},
                    geometry_pass={str(t):float(np.mean(np.array(residuals)<=t)) if residuals else None for t in TOLERANCES},
                    cpu_ms_per_pair=seconds*1000/len(pred),special_validation=by_category,
                    data_quality=dict(ordinary=quality(data),special=quality(special)),epoch_selections=epoch_rows,
                    python=platform.python_version(),numpy=np.__version__,
                    dataset_sha256=hashlib.sha256((ROOT/'data/experiment_large.npz').read_bytes()).hexdigest(),
                    protocol='original 400-pair validation; special 210-pair validation reported separately; no test evaluation')
        (out/'predictions.json').write_text(json.dumps([np.asarray(p).tolist() for p in pred]),encoding='utf-8')
    (out/'evaluation.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print('evaluation_file='+str(out/'evaluation.json'),flush=True)
    return result


def guard():
    result=evaluate()
    assert result['guard_pass']
    config=result['config']
    model=IntegratedIntersectionMLP(config,dict(np.load(ROOT/'.tmp/autoresearch_nurbs'/result['key']/'weights.npz')))
    data=load_validation()
    q,s,_=model(data['curve_a'][0],data['curve_b'][0])
    assert q.shape==(8,2) and s.shape==(8,) and np.all(np.isfinite(q)) and np.all((q>=0)&(q<=1))
    assert result['metrics']['0.035']['f1']>=.88
    print('guard_pass')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--guard',action='store_true');args=parser.parse_args()
    if args.guard:guard()
    else:
        result=evaluate();print(json.dumps(dict(score=result['score'],f1_035=result.get('metrics',{}).get('0.035',{}).get('f1'),key=result['key'])),flush=True)

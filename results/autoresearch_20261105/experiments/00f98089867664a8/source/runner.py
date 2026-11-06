"""Training only: protocol, labels, tolerances and checkpoint choice are frozen outside scope."""
import hashlib
import json
import os
from pathlib import Path
import numpy as np
from code.experiments import learned_correction as lc
from code.experiments.autoresearch_nurbs.model import IntegratedIntersectionMLP, prepare

ROOT = Path(__file__).resolve().parents[3]


def config_and_key():
    directory = Path(__file__).parent
    config = json.loads((directory/'config.json').read_text(encoding='utf-8'))
    digest = hashlib.sha256()
    for path in sorted(directory.glob('*.py')) + [directory/'config.json']:
        digest.update(path.name.encode()); digest.update(path.read_bytes())
    for path in (ROOT/'data/experiment_large.npz',ROOT/'results/learned_study/refresh_scale_0p04/weights.npz',
                 ROOT/'code/evaluation/autoresearch_nurbs_protocol.py',ROOT/'code/experiments/learned_correction.py',
                 ROOT/'code/experiments/candidate_model.py',ROOT/'data/meeting_analytic_validation.npz'):
        digest.update(path.read_bytes())
    return config, digest.hexdigest()[:16]


def train_or_load():
    config,key = config_and_key()
    out = ROOT/'.tmp/autoresearch_nurbs'/key
    out.mkdir(parents=True,exist_ok=True)
    (out/'source').mkdir(exist_ok=True)
    for path in Path(__file__).parent.iterdir():
        if path.suffix in ('.py','.json'):
            (out/'source'/path.name).write_bytes(path.read_bytes())
    if (out/'training_complete.json').exists():
        return config,key,out
    state = dict(np.load(ROOT/'results/learned_study/refresh_scale_0p04/weights.npz'))
    model = IntegratedIntersectionMLP(config,state)
    if config['reuse_pretrained']:
        np.savez_compressed(out/'epoch_000.npz',**state)
        history = []
    else:
        archive = np.load(ROOT/'data/experiment_large.npz')
        a,b,labels,counts = (archive['train_'+name] for name in ('curve_a','curve_b','labels','counts'))
        if config['analytic_training']:
            analytic = np.load(ROOT/'data/meeting_analytic_train.npz')
            a=np.concatenate([a,analytic['curve_a']]); b=np.concatenate([b,analytic['curve_b']])
            labels=np.concatenate([labels,analytic['labels']]); counts=np.concatenate([counts,analytic['counts']])
        data = prepare(a,b,labels,counts,config)
        rng = np.random.default_rng(config['seed'])
        history=[]; iteration=0
        for epoch in range(1,config['epochs']+1):
            order=rng.permutation(len(data['target'])); sums=np.zeros(4); seen=0
            for start in range(0,len(order),config['batch_size']):
                ids=order[start:start+config['batch_size']]
                values=lc.batch_loss_and_gradient(model.solver,data['features'][ids],data['target'][ids],
                    data['curve_a'][ids],data['curve_b'][ids],config['steps'],config['geometry_weight'],
                    config['regression_weight'],config['positive_weight'],config['deep_supervision'])
                iteration+=1
                model.solver.network.step(config['learning_rate'],.9,.999,1e-8,iteration)
                sums+=np.asarray(values)*len(ids); seen+=len(ids)
            row=dict(epoch=epoch,train_loss=float(sums[0]/seen),classification=float(sums[1]/seen),
                     parameter=float(sums[2]/seen),geometry=float(sums[3]/seen))
            history.append(row)
            print(json.dumps(row),flush=True)
            if epoch%5==0 or epoch==config['epochs']:
                np.savez_compressed(out/f'epoch_{epoch:03d}.npz',**model.solver.state())
            (out/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    (out/'training_complete.json').write_text(json.dumps(dict(key=key,epochs=len(history))),encoding='utf-8')
    return config,key,out

"""Reproduces the Phase A table in docs/RESEARCH_AND_PLAN_V2.md.

Usage:  python -m atdrps.cli corpus --captures 64 --duration 3600 --campaigns 3 --out DIR/corpus_v1sim.npz --workers 4
        python scripts/forecast_baselines_v1sim.py DIR
"""
import sys, numpy as np, time
sys.path.insert(0,'.')
from atdrps.cli import _load_corpus
from pathlib import Path
from atdrps.forecast.protocol import *
from atdrps.forecast.baselines import *
S=sys.argv[1]
caps=_load_corpus(Path(S+'/corpus_v1sim.npz'))
print('captures',len(caps),'windows',sum(len(c) for c in caps),'attack-window share',
      round(np.mean(np.concatenate([c.infiltration for c in caps])),3))
C,K=16,5
samples=build_onset_samples(caps,context=C,horizon=K,quiet_gap=3)
print(samples.describe())
tr,va,te=split_groups(samples,0.2,0.3,seed=0)
print('train',tr.describe()); print('val  ',va.describe()); print('test ',te.describe())
train_g=np.unique(tr.group); 
models=[PriorBaseline(),CusumBaseline(),MahalanobisBaseline(),IsolationForestBaseline()]
rows=[]
for m in models:
    m.fit(tr); r=evaluate(te,m.score(te.context),val_scores=m.score(va.context),val_samples=va,n_boot=200)
    rows.append((m.name,r))
v1=V1Baseline(C,K).fit_captures([caps[g] for g in np.unique(np.concatenate([tr.group,va.group]))])
r=evaluate(te,v1.score(te.context),val_scores=v1.score(va.context),val_samples=va,n_boot=200); rows.append((v1.name,r))
print(f"\n{'model':38s} AUC [95% CI]           AP(chance) TPR@1%FPR  eventRecall lead(s)  PPV@1e-3")
for n,r in rows:
    o=r['operating']; lo,hi=r['ci95']['auc']
    print(f"{n:38s} {r['auc']:.3f} [{lo:.3f},{hi:.3f}]  {r['ap']:.3f}({r['chance_ap']:.3f})  {r['tpr_at_fpr_test']['0.01']:.3f}      {o['event_recall']:.3f}       {o['median_lead_s']:.0f}      {o['ppv@0.001']:.4f}")
print('negatives in test:',rows[0][1]['n_neg'],' resolvable 1% budget:',rows[0][1]['budget_resolvable'],' onsets:',rows[0][1]['operating']['n_events'])

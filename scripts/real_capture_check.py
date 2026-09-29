"""Runs the v1 engine on real and crafted captures and measures how far they sit from the simulator.

Usage: python scripts/real_capture_check.py SCRATCH_DIR CAPTURE_DIR   (see docs/RESEARCH_AND_PLAN_V2.md)
"""
import sys, numpy as np, collections
sys.path.insert(0,'.')
from pathlib import Path
from atdrps.engine.inference import ThreatForecastEngine, load_capture
from atdrps.cli import _load_corpus
S=sys.argv[1]; U=sys.argv[2].rstrip("/")+"/"   # S: dir with model-v1/ and corpus_v1sim.npz ; U: dir with the captures
FILES=[("REAL laptop, 145 min (your screenshot)","9b8eaf2c-3rr.pcapng"),
       ("REAL laptop, 24 s","04b4b164-1.pcapng"),
       ("crafted 'benign_safe'","0d88f98d-ATDRPS_benign_safe_capture.pcap"),
       ("crafted 'preattack_threshold_085'","fe07406a-ATDRPS_preattack_threshold_085.pcap")]
bg=np.load(S+'/model-v1/background.npy') if Path(S+'/model-v1/background.npy').exists() else None
eng=ThreatForecastEngine.load(S+'/model-v1',background=bg,threshold=0.85)
caps=_load_corpus(Path(S+'/corpus_v1sim.npz'))
Xtr=np.concatenate([c.X[c.infiltration==0] for c in caps]); mu,sd=Xtr.mean(0),Xtr.std(0)+1e-6
lo,hi=np.percentile(Xtr,0.5,axis=0),np.percentile(Xtr,99.5,axis=0)
print(f"{'capture':40s} win  raw-prob mean/p90/max   >=0.85   CONFIRMED  stage-mix(top)                 median|z|  windows w/ >=20 feats outside sim range")
for label,f in FILES:
    try: res=eng.analyse(U+f,explain=False)
    except Exception as e: print(f"{label:40s} error {e}"); continue
    states,_,_=load_capture(U+f,30.0)
    if not res.timeline:
        Xr=states.X; z=(Xr-mu)/sd; out=((Xr<lo)|(Xr>hi)).sum(1)
        print(f"{label:40s} {len(states):3d}  (too short for the 16-window context)     median|z|={np.median(np.abs(z)):.2f}  {100*np.mean(out>=20):.0f}%"); continue
    p=np.array([r['infiltration_probability'] for r in res.timeline]); lv=collections.Counter(r['level'] for r in res.timeline)
    st=collections.Counter(r['stage'] for r in res.timeline).most_common(2)
    Xr=states.X; z=(Xr-mu)/sd; out=((Xr<lo)|(Xr>hi)).sum(1)
    print(f"{label:40s} {len(states):3d}  {p.mean():.2f}/{np.percentile(p,90):.2f}/{p.max():.2f}        {int((p>=.85).sum()):3d}      {lv.get('CONFIRMED',0):3d}       {st}  {np.median(np.abs(z)):.2f}      {100*np.mean(out>=20):.0f}%")
print('\nreference: simulator benign windows median|z| = %.2f'%np.median(np.abs((Xtr-mu)/sd)))

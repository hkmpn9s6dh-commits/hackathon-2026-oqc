# -*- coding: utf-8 -*-
"""Evaluate the decoder ladder on seeds the hyper-parameters never saw."""
import sys, os, json, time
import numpy as np
HERE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(HERE,'src'))
from oqc import metrics, decoders, dem as dm

dem_path = os.path.join(HERE,'data','qec','dem.txt')
shotted = int(sys.argv[1]) if len(sys.argv)>1 else 10000
rows=[]
for seed in (202, 303, 404, 505):
    det, obs = metrics.sample(shots=shotted, seed=seed)
    line={'seed':seed}
    t=time.time(); base=decoders.MWPM(dm.load_dem_text(dem_path)); p=base.decode_obs(det); line['mwpm']=float((p!=obs).mean())
    corr=decoders.MWPM(dm.load_dem_text(dem_path), correlations=True); p=corr.decode_obs(det); line['mwpm_corr']=float((p!=obs).mean())
    ens=decoders.EnsembleMWPM(dm.load_dem_text(dem_path), shots_per_round=120, temperature=0.75, seed=99)
    p=ens.decode_obs(det); line['ensemble']=float((p!=obs).mean()); line['ensemble_errors']=int((p!=obs).sum())
    hy=decoders.HybridEnsemble(dm.load_dem_text(dem_path), margin=0.5,
        ensemble_kwargs=dict(shots_per_round=120, temperature=0.75, seed=99),
        bposd_kwargs=dict(max_iter=60, osd_method='osd_cs', osd_order=20))
    t=time.time(); p=hy.decode_obs(det); dt=time.time()-t
    line['hybrid']=float((p!=obs).mean()); line['hybrid_errors']=int((p!=obs).sum()); line['hybrid_seconds']=round(dt,1)
    line['low_margin']=hy.last_diagnostics
    lo,hi=metrics.wilson(line['hybrid_errors'], shotted); line['ci95']=[lo,hi]
    print(json.dumps(line, ensure_ascii=False), flush=True)
    rows.append(line)
json.dump(rows, open(os.path.join(HERE,'out','validate_hybrid.json'),'w'), indent=2)

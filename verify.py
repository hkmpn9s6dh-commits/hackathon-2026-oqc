# -*- coding: utf-8 -*-
"""Assertions for the OQC entry. Exit code is the verdict: 0 = all green.

    python verify.py

Nothing here is a print-and-eyeball check: every claim in docs/04-RESULTS.md that can be
mechanically tested is tested here.  Checks fall into five groups: the data we were given, the
identity of the DEM, the metric's calibration against the organiser's own anchor, the ordering
of the decoders on identical shots, and the shape/validity of the submission files.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import stim

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from oqc import decoders, dem as dm, metrics  # noqa: E402

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print("%s  %-58s %s" % ("PASS" if cond else "FAIL", name, detail))
    return bool(cond)


def main():
    dem_path = os.path.join(HERE, "data", "qec", "dem.txt")
    det_path = os.path.join(HERE, "data", "qec", "detectors.npy")

    # ---------------------------------------------------------------- 1. the given data
    det = np.load(det_path)
    check("given detectors shape is (10000, 120)", det.shape == (10000, 120), str(det.shape))
    check("given detectors are 0/1", set(np.unique(det).tolist()) <= {0, 1}, str(np.unique(det)))
    check("given defect rate in [0.060, 0.080]", 0.060 <= det.mean() <= 0.080, "%.5f" % det.mean())

    # ---------------------------------------------------------------- 2. DEM identity
    given = dm.load_dem_text(dem_path)
    frozen = dm.frozen_dem(0.005, 5, decompose=True)
    check("dem.txt is byte-identical to the frozen circuit's decomposed DEM", str(given) == str(frozen))
    check("DEM has 120 detectors, 1 observable", (given.num_detectors, given.num_observables) == (120, 1),
          "%d/%d" % (given.num_detectors, given.num_observables))
    check("decomposed DEM has 1953 mechanisms", given.num_errors == 1953, str(given.num_errors))
    und = dm.frozen_dem(0.005, 5, decompose=False)
    check("undecomposed DEM has 1677 mechanisms (the board's 'merged' count)", und.num_errors == 1677,
          str(und.num_errors))

    h, obs_vec, probs = dm.dem_to_matrix(und)
    check("matrix shape matches DEM", h.shape == (120, 1677), str(h.shape))
    check("every mechanism keeps 1-4 detectors", len(np.unique(h.tocsc().indptr)) > 1
          and int(np.diff(h.tocsc().indptr).max()) == 4,
          "max %d" % int(np.diff(h.tocsc().indptr).max()))
    check("probabilities are inside (0, 0.5)", bool((probs > 0).all() and (probs < 0.5).all()),
          "[%.2e, %.4f]" % (probs.min(), probs.max()))
    n_obs = int(obs_vec.sum())
    check("observable-flipping mechanisms are a minority", 0 < n_obs < len(obs_vec) // 2, "%d/%d" % (n_obs, len(obs_vec)))
    check("no mechanism flips the observable without touching a detector",
          not bool(((h.getnnz(axis=0) == 0) & (obs_vec == 1)).any()))

    # ---------------------------------------------------------------- 3. metric + calibration
    lo, hi = metrics.wilson(0, 100)
    check("wilson(0,100) has lo=0 and a positive hi", lo == 0.0 and 0 < hi < 1, "[%.4f, %.4f]" % (lo, hi))
    lo2, hi2 = metrics.wilson(50, 100)
    check("wilson is symmetric around 0.5", abs((lo2 + hi2) / 2 - 0.5) < 1e-9, "[%.4f, %.4f]" % (lo2, hi2))
    check("wilson narrows as n grows", (metrics.wilson(80, 10000)[1] - metrics.wilson(80, 10000)[0]) <
          (metrics.wilson(8, 1000)[1] - metrics.wilson(8, 1000)[0]))

    # the organiser's seed-13 sampler must NOT reproduce the published shots
    frozen_det, _ = metrics.sample(shots=10000, seed=13)
    check("public 10k shots are NOT the organiser's frozen sampler prefix",
          not bool((frozen_det.astype(np.uint8) == det.astype(np.uint8)).all()),
          "answer key not derivable from the public seed")

    # ---------------------------------------------------------------- 4. decoders on identical shots
    det_bench, obs_bench = metrics.sample(shots=3000, seed=101)
    plain = decoders.MWPM(dm.load_dem_text(dem_path))
    corr = decoders.MWPM(dm.load_dem_text(dem_path), correlations=True)
    p_plain = plain.decode_obs(det_bench)
    p_corr = corr.decode_obs(det_bench)
    e_plain = int((p_plain != obs_bench).sum())
    e_corr = int((p_corr != obs_bench).sum())
    check("both decoders emit one bit per shot", p_plain.shape == p_corr.shape == (3000,))
    check("correlated matching is at least as good as plain matching on identical shots",
          e_corr <= e_plain, "%d vs %d errors" % (e_corr, e_plain))
    lo, hi = metrics.wilson(e_plain, len(obs_bench))
    check("plain MWPM reproduces the organiser's 0.0153 anchor inside its 95% CI",
          lo <= 0.0153 <= hi, "[%.4f, %.4f]" % (lo, hi))

    # BP+OSD must at least satisfy the syndrome it was handed
    bp = decoders.BPOSD(h, obs_vec, probs, max_iter=30, osd_method="osd_cs", osd_order=10)
    mism = 0
    for s in det_bench[:150]:
        corr_bits = bp.decode_correction(s)
        if int(((h @ corr_bits.astype(np.int64)) % 2 - s.astype(np.int64)).__abs__().sum()):
            mism += 1
    check("BP+OSD corrections always reproduce the syndrome", mism == 0, "%d mismatches/150" % mism)

    hy = decoders.HybridEnsemble(
        dm.load_dem_text(dem_path), margin=0.5,
        ensemble_kwargs=dict(shots_per_round=24, temperature=0.75, seed=99),
        bposd_kwargs=dict(max_iter=60, osd_method="osd_cs", osd_order=20),
    )
    p_hy = hy.decode_obs(det_bench)
    e_hy = int((p_hy != obs_bench).sum())
    check("hybrid decoder is no worse than correlated matching", e_hy <= e_corr,
          "%d vs %d errors" % (e_hy, e_corr))
    check("hybrid only escalates a small fraction of shots",
          hy.last_diagnostics["fraction"] < 0.05, str(hy.last_diagnostics))

    # ---------------------------------------------------------------- 5. submission validity
    sub_path = os.path.join(HERE, "out", "predictions_qec.json")
    with open(sub_path, "r", encoding="utf-8") as fh:
        sub = json.load(fh)
    preds = sub["predictions"]
    check("submission has all 10000 shot ids", len(preds) == 10000, str(len(preds)))
    check("shot ids are zero-padded and complete",
          all("shot_%05d" % i in preds for i in range(10000)))
    vals = set(preds.values())
    check("all predictions are 0/1 ints", vals <= {0, 1}, str(sorted(vals)))
    check("method description present and within the Space's 200-char cap",
          bool(sub.get("method")) and len(sub["method"]) <= 200, "%d chars" % len(sub.get("method", "")))
    ones = sum(preds.values())
    check("predicted ones within a plausible band", 1500 <= ones <= 3200, str(ones))
    bits = np.array([preds["shot_%05d" % i] for i in range(10000)], dtype=np.uint8)
    import hashlib
    digest = hashlib.sha256(bits.tobytes()).hexdigest()
    side = os.path.join(HERE, "out", "predictions_qec.sha256")
    recorded = open(side, "r", encoding="utf-8").read().strip() if os.path.exists(side) else ""
    check("submission bit-array digest matches the recorded sidecar", digest == recorded,
          "%s vs %s" % (digest[:16], recorded[:16]))

    # ---------------------------------------------------------------- 6. Season 1 audit
    qs_path = os.path.join(HERE, "out", "qsim_energies.json")
    with open(qs_path, "r", encoding="utf-8") as fh:
        qs = json.load(fh)
    check("QSIM has 9 systems", len(qs["systems"]) == 9, str(len(qs["systems"])))
    check("QSIM energies are finite and negative",
          all(np.isfinite(v["energy"]) and v["energy"] < 0 for v in qs["systems"].values()))
    two = qs["systems"]["hub_1x2_U4"]["energy"]
    analytic = (4.0 - math.sqrt(4.0 ** 2 + 16.0)) / 2.0
    check("2-site energy matches the closed form (U - sqrt(U^2+16t^2))/2",
          abs(two - analytic) < 1e-9, "%.9f vs %.9f" % (two, analytic))
    check("QSIM total is recorded and reproducible",
          abs(qs["total"] - sum(v["energy"] for v in qs["systems"].values())) < 1e-9,
          "%.6f" % qs["total"])
    check("QSIM total disagrees with the published floor by >1 (documented audit finding)",
          abs(qs["total"] - (-36.4448)) > 1.0, "diff %.4f" % (qs["total"] + 36.4448))

    print("\n%d passed, %d failed" % (len(PASS), len(FAIL)))
    if FAIL:
        print("FAILED: " + "; ".join(FAIL))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

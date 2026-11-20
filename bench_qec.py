# -*- coding: utf-8 -*-
"""Benchmark one decoder on freshly sampled shots.

    python bench_qec.py --decoder mwpm  --shots 20000 --seed 101
    python bench_qec.py --decoder bposd --dem undecomposed --shots 20000 --seed 101

Every run prints the logical error rate, its 95% Wilson interval, wall time, and the sha256 of
the prediction bits, so a number can be traced back to an exact artifact.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from oqc import decoders, dem as demmod, metrics  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "qec", "dem.txt")


def build(decoder: str, which: str, args):
    if which == "decomposed":
        d = demmod.load_dem_text(DATA)
    else:
        d = demmod.frozen_dem(0.005, 5, decompose=False)
    if decoder in ("mwpm", "mwpm_corr"):
        return decoders.MWPM(d, correlations=(decoder == "mwpm_corr")), d
    h, obs, probs = demmod.dem_to_matrix(d)
    return decoders.BPOSD(
        h, obs, probs,
        max_iter=args.max_iter,
        bp_method=args.bp_method,
        ms_scaling_factor=args.ms_scaling,
        osd_method=args.osd_method,
        osd_order=args.osd_order,
        name="bposd_%s_%s_it%d_%s%d" % (which, args.bp_method, args.max_iter, args.osd_method, args.osd_order),
    ), d


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--decoder", choices=["mwpm", "mwpm_corr", "bposd"], default="mwpm")
    ap.add_argument("--dem", choices=["decomposed", "undecomposed"], default="decomposed")
    ap.add_argument("--shots", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=101)
    ap.add_argument("--p", type=float, default=0.005)
    ap.add_argument("--d", type=int, default=5)
    ap.add_argument("--max-iter", type=int, default=60)
    ap.add_argument("--bp-method", default="ps", choices=["ps", "ms", "msl"])
    ap.add_argument("--ms-scaling", type=float, default=1.0)
    ap.add_argument("--osd-method", default="osd_cs", choices=["osd_0", "osd_e", "osd_cs"])
    ap.add_argument("--osd-order", type=int, default=10)
    ap.add_argument("--out", default="")
    ap.add_argument("--save-pred", default="")
    args = ap.parse_args(argv)

    det, obs = metrics.sample(p=args.p, d=args.d, shots=args.shots, seed=args.seed)
    print("shots %d  seed %d  defect rate %.5f  raw observable flip rate %.4f"
          % (args.shots, args.seed, det.mean(), obs.mean()), flush=True)

    t0 = time.time()
    decoder, d = build(args.decoder, args.dem, args)
    if hasattr(decoder, "decode_obs"):
        pred = decoder.decode_obs(det)
    else:
        pred = decoder.decode_obs_batch(det, verbose=True)
    dt = time.time() - t0

    err = int((pred != obs).sum())
    ler = err / len(obs)
    lo, hi = metrics.wilson(err, len(obs))
    digest = hashlib.sha256(np.ascontiguousarray(pred, dtype=np.uint8).tobytes()).hexdigest()
    print("%-46s LER %.5f  95%%CI [%.5f, %.5f]  errors %d/%d  %.1fs  sha256 %s"
          % (decoder.name, ler, lo, hi, err, len(obs), dt, digest[:16]), flush=True)

    if args.save_pred:
        np.savez_compressed(args.save_pred, pred=pred, obs=obs, seed=args.seed, decoder=decoder.name)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({
                "decoder": decoder.name, "dem": args.dem, "shots": args.shots, "seed": args.seed,
                "p": args.p, "d": args.d, "errors": err, "ler": ler, "ci95": [lo, hi],
                "seconds": round(dt, 2), "sha256": digest,
            }, fh, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

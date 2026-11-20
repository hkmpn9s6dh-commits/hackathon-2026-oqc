# -*- coding: utf-8 -*-
"""Produce the two submission files.

    python make_submission.py            # QEC (Season 2) + QSIM (Season 1)

Outputs, both in the JSON shape the Space's /api/submit endpoint accepts:

    out/predictions_qec.json   {"hf_user", "method", "predictions": {"shot_00000": 0/1, ...}}
    out/predictions_qsim.json  {"hf_user", "method", "predictions": {"hub_1x2_U4": -0.8284..., ...}}

The QEC file is the real entry.  The QSIM file is a by-product of the exact diagonalisation
used to audit the Season-1 floor; see docs/04-RESULTS.md for why it is not recommended for
submission (that board is saturated and its published floor does not agree with the stated
Hamiltonian).
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from oqc import decoders, dem as dm  # noqa: E402

DEM = os.path.join(HERE, "data", "qec", "dem.txt")
DETECTORS = os.path.join(HERE, "data", "qec", "detectors.npy")
OUT = os.path.join(HERE, "out")

METHOD_QEC = (
    "Ensemble correlated-MWPM: 120 jittered log-odds draws, temperature-0.75 weighted vote; "
    "BP+OSD (undecomposed 1677-fault DEM) breaks ties when the vote margin is below 0.5."
)


def sha256_bits(pred: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(pred, dtype=np.uint8).tobytes()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    det = np.load(DETECTORS).astype(np.uint8)
    assert det.shape == (10000, 120), det.shape

    t0 = time.time()
    hy = decoders.HybridEnsemble(
        dm.load_dem_text(DEM),
        margin=0.5,
        ensemble_kwargs=dict(shots_per_round=120, temperature=0.75, seed=99),
        bposd_kwargs=dict(max_iter=60, osd_method="osd_cs", osd_order=20),
    )
    pred = hy.decode_obs(det)
    dt = time.time() - t0

    payload = {
        "hf_user": "hkmpn9s6dh-commits",
        "method": METHOD_QEC,
        "predictions": {"shot_%05d" % i: int(v) for i, v in enumerate(pred)},
    }
    path = os.path.join(OUT, "predictions_qec.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    digest = sha256_bits(pred)
    with open(os.path.join(OUT, "predictions_qec.sha256"), "w", encoding="utf-8") as fh:
        fh.write(digest + "\n")
    print("qec: %d predictions, ones %d, %.1fs, sha256 %s -> %s"
          % (len(pred), int(pred.sum()), dt, digest[:16], path))
    print("qec diagnostics:", hy.last_diagnostics)

    # Season 1 file (audit by-product)
    qsim_path = os.path.join(OUT, "qsim_energies.json")
    if os.path.exists(qsim_path):
        with open(qsim_path, "r", encoding="utf-8") as fh:
            qsim = json.load(fh)
        qsim_payload = {
            "hf_user": "hkmpn9s6dh-commits",
            "method": "Exact diagonalisation (sparse Lanczos per (N_up,N_dn) sector, open BC, "
                      "H=-t sum c^dag c + U sum n_up n_dn). Audit by-product, not recommended.",
            "predictions": {k: round(v["energy"], 9) for k, v in qsim["systems"].items()},
        }
        p2 = os.path.join(OUT, "predictions_qsim.json")
        with open(p2, "w", encoding="utf-8") as fh:
            json.dump(qsim_payload, fh, indent=2)
        print("qsim: %d systems, total %.6f -> %s" % (len(qsim_payload["predictions"]),
                                                     sum(qsim_payload["predictions"].values()), p2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

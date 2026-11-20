# -*- coding: utf-8 -*-
# Vendored verbatim from the organiser Space (FINAL-Bench/open-quantum-challenge, MIT)
# file gates.py @ sha 2dd338be, fetched 2026-11-21, so verify.py can run the *official*
# shape check offline instead of trusting our own re-implementation.
# -*- coding: utf-8 -*-
"""Submission gates for OQC. No chemistry here — a quantum entry is a prediction file.

The gate only checks *shape*: that the payload is the right kind of thing for the board, so
the scoring worker is never handed garbage. The authoritative scoring (and leakage / exact-hit
accounting) stays with the worker, which holds the private answer key.
"""
import math

# generous caps — QEC has 10k shots; keep a ceiling so one request cannot be a megabyte bomb
MAX_ITEMS = 60000
MAX_BYTES = 6_000_000


def check(board, predictions):
    reject = []
    if not isinstance(predictions, dict) or not predictions:
        return {"admitted": False,
                "reject": ["predictions must be a non-empty JSON object { item_id: value }"]}
    if len(predictions) > MAX_ITEMS:
        return {"admitted": False,
                "reject": ["too many items (%d > %d)" % (len(predictions), MAX_ITEMS)]}

    bad = 0
    if board == "qec":
        for k, v in predictions.items():
            try:
                iv = int(v)
            except (TypeError, ValueError):
                bad += 1
                continue
            if iv not in (0, 1):
                bad += 1
        if bad:
            reject.append("QEC predictions must be 0 or 1 per shot (%d invalid)" % bad)
        # shot ids look like shot_00000 ; we don't require completeness (worker counts misses)
    elif board == "qsim":
        for k, v in predictions.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                bad += 1
                continue
            if not math.isfinite(fv):
                bad += 1
        if bad:
            reject.append("QSIM predictions must be finite numbers (energies) (%d invalid)" % bad)
    else:
        reject.append("unknown board %r" % board)

    return {"admitted": not reject, "reject": reject, "n_items": len(predictions)}

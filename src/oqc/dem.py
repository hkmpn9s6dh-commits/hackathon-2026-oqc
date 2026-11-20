# -*- coding: utf-8 -*-
"""Parsing the public detector error model into the two shapes we need.

Two representations matter for this challenge, and confusing them costs real accuracy:

  * the *decomposed* DEM shipped as ``qec/dem.txt`` (1953 mechanisms) — each circuit fault
    that flips three or four detectors is split into two graphlike pieces, so a matcher can
    consume it.  The split doubles the number of independent binary variables and therefore
    pays a probability penalty;
  * the *undecomposed* DEM (1677 mechanisms, one per actual circuit fault) — physically what
    the hardware does, and the shape the top of the leaderboard reports using ("1677 merged
    mechanisms").  A belief-propagation decoder is happier here because a Y error stays one
    variable instead of two weakly-tied halves.

``stim`` will regenerate both from the frozen circuit, and the decomposed one is byte-identical
to the file published in ``FINAL-Bench/oqc-data`` (asserted in verify.py).
"""
from __future__ import annotations

import numpy as np
import stim
from scipy.sparse import csr_matrix


def load_dem_text(path: str) -> stim.DetectorErrorModel:
    with open(path, "r", encoding="utf-8") as fh:
        return stim.DetectorErrorModel(fh.read())


def frozen_dem(p: float, d: int, decompose: bool) -> stim.DetectorErrorModel:
    """The DEM of the exact circuit the organiser scores against."""
    circuit = stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )
    return circuit.detector_error_model(decompose_errors=decompose)


def dem_to_matrix(dem: stim.DetectorErrorModel):
    """Flatten a DEM into (H, observable vector, mechanism probabilities).

    ``H`` is detectors x mechanisms over GF(2); ``obs`` is one bit per mechanism (the parity of
    the logical observable that mechanism flips).  This is the linear code the decoders work on.
    """
    rows, cols, probs, obs = [], [], [], []
    k = 0
    for inst in dem.flattened():
        if inst.type != "error":
            # repeat blocks are already flattened; instructions carry no mechanism
            continue
        p = float(inst.args_copy()[0])
        dets, ob = [], 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(int(t.val))
            elif t.is_logical_observable_id():
                ob ^= 1
        for dd in dets:
            rows.append(dd)
            cols.append(k)
        probs.append(p)
        obs.append(ob)
        k += 1
    h = csr_matrix(
        (np.ones(len(rows), dtype=np.uint8), (np.asarray(rows, int), np.asarray(cols, int))),
        shape=(dem.num_detectors, k),
    )
    h.sum_duplicates()
    h.data[:] = 1  # GF(2): a fault that hits the same detector twice cancels
    return h, np.asarray(obs, dtype=np.uint8), np.asarray(probs, dtype=np.float64)


def mechanism_signatures(dem: stim.DetectorErrorModel):
    """Return list of (frozenset_of_detectors, obs_bit) per mechanism, in matrix order."""
    sig = []
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        dets, ob = [], 0
        for t in inst.targets_copy():
            if t.is_relative_detector_id():
                dets.append(int(t.val))
            elif t.is_logical_observable_id():
                ob ^= 1
        sig.append((tuple(sorted(set(dets))), ob))
    return sig


def merge_identical(dem: stim.DetectorErrorModel):
    """Collapse mechanisms with the same (detector set, observable) signature.

    Two independent faults with the same signature combine by XOR: p = p1(1-p2) + p2(1-p1).
    Returns a DEM-like triple plus the multiplicity, useful when reporting "how many distinct
    syndromes a decoder actually has to tell apart".
    """
    sig = mechanism_signatures(dem)
    order, acc = {}, {}
    for (dets, ob), inst in zip(sig, [i for i in dem.flattened() if i.type == "error"]):
        p = float(inst.args_copy()[0])
        key = (dets, ob)
        if key in acc:
            q = acc[key]
            acc[key] = q * (1 - p) + p * (1 - q)
            order[key] += 1
        else:
            acc[key] = p
            order[key] = 1
    return acc, order


def gf2_residual(h: csr_matrix, correction: np.ndarray, syndrome: np.ndarray) -> np.ndarray:
    """Syndrome left over by a candidate correction (empty means the correction is consistent)."""
    return (h @ correction.astype(np.int64)) % 2 - syndrome.astype(np.int64)

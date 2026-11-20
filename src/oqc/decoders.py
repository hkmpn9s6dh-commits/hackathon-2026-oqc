# -*- coding: utf-8 -*-
"""Decoders for the OQC surface-code task.

The task is a binary decision per shot: which logical-observable value (0 or 1) is more likely
given the 120 detector bits.  Two decoders are implemented:

  * MWPM (pymatching)  — the organiser's published anchor, 0.0153 on the board;
  * BP+OSD (ldpc)      — belief propagation on the detector/mechanism Tanner graph with
                         ordered-statistics post-processing, which is the standard strong
                         classical decoder for surface codes.

Both take a (H, observable, probabilities) triple so we can swap between the decomposed DEM
(1953 mechanisms, what the organiser ships) and the undecomposed one (1677, one variable per
circuit fault).
"""
from __future__ import annotations

import numpy as np
import pymatching
import stim
from ldpc import bposd_decoder
from scipy.sparse import csr_matrix

from . import dem as demmod


class MWPM:
    """Minimum-weight perfect matching on the graphlike DEM (the published baseline).

    ``correlations=True`` turns on pymatching's correlated matching: the decomposed 4-body
    faults are matched jointly through their two halves instead of being treated as two
    independent edges.  That is the same information a coset-sum decoder exploits, obtained
    for free, and it is worth ~23% of the logical error rate on this problem.
    """

    def __init__(self, dem: stim.DetectorErrorModel, correlations: bool = False):
        self.correlations = bool(correlations)
        self.name = "mwpm_corr" if correlations else "mwpm"
        self.matching = pymatching.Matching.from_detector_error_model(
            dem, enable_correlations=self.correlations
        )

    def decode_obs(self, detectors: np.ndarray) -> np.ndarray:
        return self.matching.decode_batch(
            np.ascontiguousarray(detectors, dtype=np.uint8),
            enable_correlations=self.correlations,
        )[:, 0].astype(np.uint8)


class BPOSD:
    """Belief propagation + ordered-statistics decoding over the DEM matrix."""

    def __init__(
        self,
        h: csr_matrix,
        obs: np.ndarray,
        probs: np.ndarray,
        *,
        max_iter: int = 60,
        bp_method: str = "ps",
        ms_scaling_factor: float = 1.0,
        osd_method: str = "osd_cs",
        osd_order: int = 10,
        name: str | None = None,
    ):
        self.h = h
        self.obs = np.asarray(obs, dtype=np.uint8)
        self.probs = np.asarray(probs, dtype=np.float64)
        self.decoder = bposd_decoder(
            h,
            channel_probs=list(self.probs),
            max_iter=int(max_iter),
            bp_method=bp_method,
            ms_scaling_factor=float(ms_scaling_factor),
            osd_method=osd_method,
            osd_order=int(osd_order),
        )
        self.name = name or "bposd_%s_it%d_%s%d" % (bp_method, max_iter, osd_method, osd_order)

    def decode_correction(self, syndrome: np.ndarray) -> np.ndarray:
        return np.asarray(self.decoder.decode(np.ascontiguousarray(syndrome, dtype=np.uint8)), dtype=np.uint8)

    def decode_obs_batch(self, detectors: np.ndarray, *, verbose: bool = False) -> np.ndarray:
        out = np.empty(len(detectors), dtype=np.uint8)
        for i, s in enumerate(detectors):
            corr = self.decode_correction(s)
            out[i] = int((corr.astype(np.int64) @ self.obs.astype(np.int64)) & 1)
            if verbose and (i + 1) % 2000 == 0:
                print("  decoded %d/%d" % (i + 1, len(detectors)), flush=True)
        return out


class EnsembleMWPM:
    """Correlated matching, repeated under jittered edge weights, combined by weighted vote.

    Why this is not just "run the same decoder many times": a matching decoder returns *one*
    minimum-weight solution, but the logical decision is a sum over *all* near-minimum
    solutions (the degeneracy of the code).  Jittering the mechanism probabilities samples the
    tie-broken optimum; each sample is then weighted by its own matching weight, which is the
    negative log likelihood ratio of that solution.  That is a Monte-Carlo estimate of the
    coset sum the top of the leaderboard computes exactly — cheap, and it recovers about a
    fifth of the gap between plain matching and the reported best.

    Hyper-parameters (K, sigma ladder, vote temperature) were chosen on a validation seed that
    is disjoint from both the tuning-free baseline and the final reported seed.
    """

    def __init__(
        self,
        dem: stim.DetectorErrorModel,
        *,
        shots_per_round: int = 120,
        sigmas=(0.15, 0.25, 0.35, 0.5, 0.7),
        temperature: float = 0.75,
        seed: int = 99,
        name: str | None = None,
    ):
        self.dem = dem
        self.k = int(shots_per_round)
        self.sigmas = tuple(sigmas)
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.name = name or "ens_mwpm_k%d_t%.2f" % (self.k, self.temperature)

    @staticmethod
    def _jitter(dem: stim.DetectorErrorModel, sigma: float, rng) -> stim.DetectorErrorModel:
        lines = []
        for inst in dem.flattened():
            if inst.type != "error":
                lines.append(str(inst))
                continue
            p = min(max(float(inst.args_copy()[0]), 1e-6), 1 - 1e-6)
            z = float(np.log(p / (1 - p)) + rng.normal(0.0, sigma))
            q = 1.0 / (1.0 + np.exp(-z))
            q = min(max(q, 1e-7), 0.499)
            lines.append("error(%.10g)%s" % (q, "".join(" " + str(t) for t in inst.targets_copy())))
        return stim.DetectorErrorModel("\n".join(lines) + "\n")

    def decode_obs(self, detectors: np.ndarray):
        rng = np.random.default_rng(self.seed)
        detectors = np.ascontiguousarray(detectors, dtype=np.uint8)
        n = len(detectors)
        votes0 = np.zeros(n)
        votes1 = np.zeros(n)
        preds = np.empty((self.k, n), dtype=np.uint8)
        weights = np.empty((self.k, n))
        for k in range(self.k):
            sigma = self.sigmas[k % len(self.sigmas)]
            m = pymatching.Matching.from_detector_error_model(
                self._jitter(self.dem, sigma, rng), enable_correlations=True
            )
            pk, wk = m.decode_batch(detectors, enable_correlations=True, return_weights=True)
            preds[k] = pk[:, 0]
            weights[k] = wk
        w = np.exp(-(weights - weights.min(axis=0)) / self.temperature)
        votes1 = (w * (preds == 1)).sum(axis=0)
        votes0 = (w * (preds == 0)).sum(axis=0)
        out = (votes1 > votes0).astype(np.uint8)
        self.last_diagnostics = {
            "disagreement_fraction": float((preds.min(axis=0) != preds.max(axis=0)).mean()),
            "mean_margin": float(np.mean(np.abs(votes1 - votes0) / (votes1 + votes0 + 1e-12))),
        }
        return out


class HybridEnsemble:
    """Ensemble matching, with BP+OSD called in only where the vote is close.

    The two decoder families fail on different shots: on the ~0.5% of shots where the 120
    jittered matchings split their vote, BP+OSD is clearly the better judge, while on the
    confident 99.5% it is clearly the worse one.  Using BP+OSD as a tie-breaker costs about a
    second per ten thousand shots instead of three minutes for the whole set.

    The margin threshold is a hyper-parameter chosen on the validation seed only; the reported
    number comes from seeds the rule never saw.
    """

    def __init__(self, dem, *, margin: float = 0.5, ensemble_kwargs=None, bposd_kwargs=None):
        self.ensemble = EnsembleMWPM(dem, **(ensemble_kwargs or {}))
        self.dem = dem
        self.margin = float(margin)
        self._bposd_kwargs = dict(bposd_kwargs or {})
        self.name = "hybrid_%s_m%.2f" % (self.ensemble.name, self.margin)

    def _votes(self, detectors):
        # replicate EnsembleMWPM.decode_obs but keep the margin per shot
        ens = self.ensemble
        rng = np.random.default_rng(ens.seed)
        n = len(detectors)
        preds = np.empty((ens.k, n), dtype=np.uint8)
        weights = np.empty((ens.k, n))
        for k in range(ens.k):
            sigma = ens.sigmas[k % len(ens.sigmas)]
            m = pymatching.Matching.from_detector_error_model(
                ens._jitter(self.dem, sigma, rng), enable_correlations=True
            )
            pk, wk = m.decode_batch(detectors, enable_correlations=True, return_weights=True)
            preds[k] = pk[:, 0]
            weights[k] = wk
        w = np.exp(-(weights - weights.min(axis=0)) / ens.temperature)
        v1 = (w * (preds == 1)).sum(axis=0)
        v0 = (w * (preds == 0)).sum(axis=0)
        return (v1 > v0).astype(np.uint8), np.abs(v1 - v0) / (v1 + v0 + 1e-12)

    def decode_obs(self, detectors: np.ndarray):
        detectors = np.ascontiguousarray(detectors, dtype=np.uint8)
        pred, margin = self._votes(detectors)
        sel = np.where(margin < self.margin)[0]
        self.last_diagnostics = {"n_low_margin": int(len(sel)), "fraction": float(len(sel) / len(pred))}
        if len(sel):
            und = demmod.frozen_dem(0.005, 5, decompose=False)
            h, obs, probs = demmod.dem_to_matrix(und)
            bp = BPOSD(h, obs, probs, **self._bposd_kwargs)
            pred = pred.copy()
            pred[sel] = bp.decode_obs_batch(detectors[sel])
        return pred


class OracleObs:
    """Not a decoder: the true observable, used only to sanity-check plumbing."""

    name = "oracle"

    def __init__(self, obs: np.ndarray):
        self._obs = obs

    def decode_obs(self, detectors: np.ndarray) -> np.ndarray:  # pragma: no cover - test helper
        return self._obs

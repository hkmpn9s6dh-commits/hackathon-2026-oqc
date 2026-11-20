# -*- coding: utf-8 -*-
"""Sample generation and the scoring metric."""
from __future__ import annotations

import math

import numpy as np
import stim


def wilson(k: int, n: int, z: float = 1.96):
    """95% Wilson score interval for a binomial proportion (same formula as the organiser)."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def frozen_circuit(p: float = 0.005, d: int = 5) -> stim.Circuit:
    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance=d,
        rounds=d,
        after_clifford_depolarization=p,
        before_round_data_depolarization=p,
        before_measure_flip_probability=p,
        after_reset_flip_probability=p,
    )


def sample(p: float = 0.005, d: int = 5, shots: int = 10000, seed: int = 0):
    """Draw (detectors, observable) pairs from the frozen circuit with an arbitrary seed.

    Seed discipline (see docs/03-METHOD.md): model selection and every quoted number use seeds
    that are NOT the organiser's frozen one.  The frozen seed only appears in a final
    "same-yardstick" cross-check, never in tuning.
    """
    circuit = frozen_circuit(p, d)
    det, obs = circuit.compile_detector_sampler(seed=seed).sample(shots, separate_observables=True)
    return np.ascontiguousarray(det, dtype=np.uint8), np.ascontiguousarray(obs[:, 0], dtype=np.uint8)


def load_public_detectors(path: str = "data/qec/detectors.npy") -> np.ndarray:
    return np.load(path).astype(np.uint8)

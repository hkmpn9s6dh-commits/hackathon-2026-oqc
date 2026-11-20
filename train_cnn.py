# -*- coding: utf-8 -*-
"""A small 3D convolutional decoder for the OQC d=5 surface-code task.

Why a neural decoder: the scoring metric is the logical error rate of a *decision*, and the
Bayes-optimal decision is a function of the 120 detector bits.  The detector layout is a
6x6x6 spacetime grid (x, y in 0..10 step 2; t in 0..5), which is exactly the shape a 3D
convolution expects.  We can draw unlimited labelled shots from the frozen circuit, so a
supervised model can be trained well past the point where classical decoders saturate.

The model is deliberately small (a few 16-channel 3x3x3 convolutions + global pooling); it has
to be retrainable on a laptop CPU and every number in the report must be reproducible from a
fixed seed.

    python train_cnn.py --train 2000000 --epochs 12
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
import sys
sys.path.insert(0, os.path.join(HERE, "src"))
from oqc import metrics  # noqa: E402

T, X, Y = 6, 6, 6


def detector_grid_index(circuit):
    """Map each detector id to its flat index in the (t, x, y) grid."""
    coords = circuit.get_detector_coordinates()
    idx = np.full(circuit.num_detectors, -1, dtype=np.int64)
    for det, (x, y, t) in coords.items():
        xi, yi, ti = int(round(x / 2)), int(round(y / 2)), int(round(t))
        idx[det] = (ti * X + xi) * Y + yi
    assert (idx >= 0).all()
    return idx


def to_grid(bits: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    """(n, 120) uint8 -> (n, 1, T, X, Y) float32."""
    n = bits.shape[0]
    out = np.zeros((n, T * X * Y), dtype=np.float32)
    out[:, idx] = bits.astype(np.float32)
    return torch.from_numpy(out.reshape(n, 1, T, X, Y))


class ConvDecoder(nn.Module):
    def __init__(self, ch: int = 16):
        super().__init__()
        def block(i, o):
            return nn.Sequential(
                nn.Conv3d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm3d(o),
                nn.ReLU(inplace=True),
            )
        self.net = nn.Sequential(block(1, ch), block(ch, ch), block(ch, ch), block(ch, ch))
        self.head = nn.Linear(ch, 1)

    def forward(self, x):
        h = self.net(x)
        h = h.mean(dim=(2, 3, 4))
        return self.head(h).squeeze(-1)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=2_000_000)
    ap.add_argument("--test", type=int, default=100_000)
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--ch", type=int, default=16)
    ap.add_argument("--train-seed", type=int, default=11)
    ap.add_argument("--test-seed", type=int, default=12)
    ap.add_argument("--out", default="out/cnn_decoder.pt")
    ap.add_argument("--report", default="out/cnn_report.json")
    args = ap.parse_args(argv)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    circuit = metrics.frozen_circuit(0.005, 5)
    idx = detector_grid_index(circuit)

    t0 = time.time()
    det, obs = metrics.sample(shots=args.train, seed=args.train_seed)
    print("train sample %d in %.1fs  defect rate %.4f" % (args.train, time.time() - t0, det.mean()), flush=True)
    y = torch.from_numpy(obs.astype(np.float32))

    det_te, obs_te = metrics.sample(shots=args.test, seed=args.test_seed)
    y_te = torch.from_numpy(obs_te.astype(np.float32))

    model = ConvDecoder(args.ch)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=args.epochs * (args.train // args.batch + 1)
    )
    loss_fn = nn.BCEWithLogitsLoss()

    n = len(y)
    best = 1.0
    hist = []
    for ep in range(1, args.epochs + 1):
        perm = torch.from_numpy(rng.permutation(n))
        model.train()
        tot, seen = 0.0, 0
        te = time.time()
        for i in range(0, n, args.batch):
            sel = perm[i:i + args.batch]
            xb, yb = to_grid(det[sel.numpy()], idx), y[sel]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            loss = loss_fn(out, yb)
            loss.backward()
            opt.step()
            sched.step()
            tot += float(loss) * len(sel)
            seen += len(sel)
        model.eval()
        with torch.no_grad():
            pv = []
            for i in range(0, len(det_te), 8192):
                pv.append(torch.sigmoid(model(to_grid(det_te[i:i + 8192], idx))))
            pred = (torch.cat(pv) > 0.5).float()
        err = int((pred != y_te).sum())
        ler = err / len(y_te)
        lo, hi = metrics.wilson(err, len(y_te))
        hist.append({"epoch": ep, "train_loss": tot / seen, "test_ler": ler, "ci95": [lo, hi]})
        print("epoch %2d  loss %.5f  test LER %.5f [%.5f, %.5f]  %.0fs"
              % (ep, tot / seen, ler, lo, hi, time.time() - te), flush=True)
        if ler < best:
            best = ler
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            torch.save({"state_dict": model.state_dict(), "ch": args.ch, "idx": idx.tolist()}, args.out)

    payload = {"train": args.train, "epochs": args.epochs, "ch": args.ch,
               "test_seed": args.test_seed, "test_shots": args.test, "history": hist,
               "best_ler": best}
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print("best test LER %.5f  weights -> %s" % (best, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

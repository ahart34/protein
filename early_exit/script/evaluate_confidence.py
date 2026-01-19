#!/usr/bin/env python3
"""
Compute loss-derived AURC / Oracle AURC / Excess AURC per layer from your saved tensor format
and write results to a CSV.

Expected torch.load(input) dict keys:
  - preds_by_layer: list[Tensor], each [N, C] (logits-like by default)
  - target: Tensor [N, C] with 0/1 and optionally NaN for unlabeled

Usage:
  python excess_aurc_loss_per_layer.py --in_pt PATH.pt --out_csv out.csv
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F



def _aurc_and_excess_from_loss_and_conf(loss: np.ndarray, conf: np.ndarray) -> Tuple[float, float, float]:
    """
    loss: shape [M], >=0 (lower is better)
    conf: shape [M], higher means "more confident"
    Returns (aurc, oracle_aurc, excess_aurc)
    """
    M = int(loss.size)
    if M == 0:
        return float("nan"), float("nan"), float("nan")

    coverage = (np.arange(1, M + 1, dtype=np.float64)) / float(M)

    # Model ordering: confidence descending
    order = np.argsort(conf)[::-1]
    loss_sorted = loss[order].astype(np.float64, copy=False)
    cum_loss = np.cumsum(loss_sorted, dtype=np.float64)
    risk = cum_loss / np.arange(1, M + 1, dtype=np.float64)
    aurc = float(np.trapz(risk, coverage))

    # Oracle ordering: true loss ascending
    loss_oracle = np.sort(loss.astype(np.float64, copy=False))
    cum_loss_oracle = np.cumsum(loss_oracle, dtype=np.float64)
    oracle_risk = cum_loss_oracle / np.arange(1, M + 1, dtype=np.float64)
    oracle_aurc = float(np.trapz(oracle_risk, coverage))

    excess = aurc - oracle_aurc
    return aurc, oracle_aurc, excess

def excess_aurc_per_layer_cl(data: Dict[str, Any], preds_key: str = "preds_by_layer", target_key: str = "target") -> Dict[int, Dict[str, float]]:
    preds_by_layer: List[torch.Tensor] = data[preds_key]
    target: torch.Tensor = data[target_key]

    if not isinstance(preds_by_layer, (list, tuple)):
        raise TypeError(f"{preds_key} must be a list/tuple, got {type(preds_by_layer)}")
    if not torch.is_tensor(target):
        raise TypeError(f"{target_key} must be a torch.Tensor, got {type(target)}")
    y_all = target[:, 0]
    labeled = ~torch.isnan(y_all)
    if int(labeled.sum().item()) == 0:
        return {}
    y = y_all[labeled].long()

    out: Dict[int, Dict[str, float]] = {}

    for layer_idx, pred in enumerate(preds_by_layer):
        pred = pred.detach().cpu()
        logits = pred[labeled]
        prob = F.softmax(logits, dim=-1)
        yhat = prob.argmax(dim=-1)
        correct = (yhat == y).float()
        conf = prob.max(dim=-1).values
        error_np = (1.0 - correct).numpy()
        conf_np = conf.numpy()
        aurc, oracle_aurc, excess = _aurc_and_excess_from_loss_and_conf(error_np, conf_np)

        out[int(layer_idx)] = {
            "n_labeled": int(error_np.size),
            "mean_loss_or_acc": float(correct.mean().item()),
            "aurc": float(aurc),
            "oracle_aurc": float(oracle_aurc),
            "excess_aurc": float(excess),
        }

    return out 



def excess_aurc_per_layer_loss_derived(
    data: Dict[str, Any],
    preds_key: str = "preds_by_layer",
    target_key: str = "target",
    pred_is_prob: bool = False,
    conf_from: str = "prob",  # "prob" or "neg_loss"
) -> Dict[int, Dict[str, float]]:
    """
    Returns a dict keyed by layer idx:
      {layer: {"n_labeled": int, "mean_loss": float, "aurc": float, "oracle_aurc": float, "excess_aurc": float}}
    """
    preds_by_layer: List[torch.Tensor] = data[preds_key]
    target: torch.Tensor = data[target_key]

    if not isinstance(preds_by_layer, (list, tuple)):
        raise TypeError(f"{preds_key} must be a list/tuple, got {type(preds_by_layer)}")
    if not torch.is_tensor(target):
        raise TypeError(f"{target_key} must be a torch.Tensor, got {type(target)}")

    labeled = ~torch.isnan(target)
    if int(labeled.sum().item()) == 0:
        return {}

    y_flat = target[labeled].float().detach().cpu()

    out: Dict[int, Dict[str, float]] = {}
    for layer_idx, pred in enumerate(preds_by_layer):
        if pred.shape != target.shape:
            raise ValueError(
                f"Layer {layer_idx} shape mismatch: pred {tuple(pred.shape)} vs target {tuple(target.shape)}"
            )

        pred_flat = pred[labeled].detach().cpu().float()

        if pred_is_prob:
            p = pred_flat.clamp(1e-7, 1 - 1e-7)
            loss_flat = F.binary_cross_entropy(p, y_flat, reduction="none")
            prob_flat = p
        else:
            loss_flat = F.binary_cross_entropy_with_logits(pred_flat, y_flat, reduction="none")
            prob_flat = torch.sigmoid(pred_flat)

        loss_np = loss_flat.numpy()
        mean_loss = float(loss_np.mean())

        if conf_from == "prob":
            conf_np = prob_flat.numpy()
        elif conf_from == "neg_loss":
            conf_np = (-loss_flat).numpy()
        else:
            raise ValueError(f"Unknown conf_from={conf_from}. Use 'prob' or 'neg_loss'.")

        aurc, oracle_aurc, excess = _aurc_and_excess_from_loss_and_conf(loss_np, conf_np)

        out[int(layer_idx)] = {
            "n_labeled": int(loss_np.size),
            "mean_loss_or_acc": mean_loss,
            "aurc": float(aurc),
            "oracle_aurc": float(oracle_aurc),
            "excess_aurc": float(excess),
        }

    return out


def write_layer_csv(out_csv: Path, by_layer: Dict[int, Dict[str, float]]) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["layer", "n_labeled", "mean_loss_or_acc", "aurc", "oracle_aurc", "excess_aurc"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for layer in sorted(by_layer.keys()):
            w.writerow({"layer": layer, **by_layer[layer]})

MODELS = ["esm", "bert", "albert"]
TASKS = ["ec", "go", "cl"]

IN_DIR = Path("results/metrics")
OUT_DIR = Path("results/confidence")



def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    n_done = 0
    n_missing = 0

    for model in MODELS:
        for task in TASKS:
            in_pt = IN_DIR / f"{model}_{task}_confidence.pt"
            out_csv = OUT_DIR / f"{model}_{task}_confidence.csv"

            if not in_pt.exists():
                print(f"[skip missing] {in_pt}")
                n_missing += 1
                continue

            data = torch.load(in_pt, map_location="cpu")

            if task == "cl":
                by_layer = excess_aurc_per_layer_cl(data)
            else:
                by_layer = excess_aurc_per_layer_loss_derived(data)

            write_layer_csv(out_csv, by_layer)
            print(f"[wrote] {out_csv}  (layers={len(by_layer)})")
            n_done += 1

    print(f"\nDone. Wrote {n_done} CSVs. Skipped {n_missing} missing .pt files.")




# def main():
#     p = argparse.ArgumentParser()
#     p.add_argument("--in_pt", required=True, help="Input .pt file saved with torch.save")
#     p.add_argument("--out_csv", required=True, help="Output CSV path")
#     args = p.parse_args()

#     data = torch.load(args.in_pt, map_location="cpu")

#     # Defaults match your earlier format: logits-like preds_by_layer + target (0/1/NaN)
#     by_layer = excess_aurc_per_layer_loss_derived(data)

#     out_path = Path(args.out_csv)
#     out_path.parent.mkdir(parents=True, exist_ok=True)

#     fieldnames = ["layer", "n_labeled", "mean_loss_or_acc", "aurc", "oracle_aurc", "excess_aurc"]
#     with open(out_path, "w", newline="") as f:
#         w = csv.DictWriter(f, fieldnames=fieldnames)
#         w.writeheader()
#         for layer in sorted(by_layer.keys()):
#             row = {"layer": layer, **by_layer[layer]}
#             w.writerow(row)

#     print(f"Wrote {len(by_layer)} layers -> {out_path}")


if __name__ == "__main__":
    main()


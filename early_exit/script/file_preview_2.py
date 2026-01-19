#!/usr/bin/env python3
"""
Infer per-property number of classes (K) for CL from a saved torch file.

Expected file format:
  torch.save({"preds_by_layer": [Tensor(N, D), ...], "target": Tensor(N, P)}, out_file)

We infer K_i for each property i from target[:, i] (ignoring NaNs):
  K_i := max(label) + 1
and report label set stats.

Usage:
  python infer_k_from_cl_dump.py /path/to/out_file.pt
"""

import argparse
import json
from typing import Any, Dict, List, Tuple

import numpy as np
import torch


def _as_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def infer_k_from_target_column(col: torch.Tensor) -> Dict[str, Any]:
    """
    col: Tensor [N] with NaNs for unlabeled
    Returns stats + inferred K (if possible).
    """
    col_cpu = col.detach().cpu()

    labeled = ~torch.isnan(col_cpu)
    n_labeled = int(labeled.sum().item())
    if n_labeled == 0:
        return {
            "n_labeled": 0,
            "unique_labels": [],
            "min_label": None,
            "max_label": None,
            "inferred_k": None,
            "notes": "No labeled entries",
        }

    vals = col_cpu[labeled]

    # Many CL targets are stored as floats but represent integer class IDs.
    # We'll check "integer-likeness".
    vals_np = _as_numpy(vals).astype(np.float64)
    frac = np.abs(vals_np - np.round(vals_np))
    max_frac = float(np.max(frac)) if frac.size else 0.0
    int_like = max_frac < 1e-6

    if not int_like:
        # If targets are not integer-like, we can't infer classes reliably.
        uniq = np.unique(vals_np)
        uniq_preview = uniq[:20].tolist()
        return {
            "n_labeled": n_labeled,
            "unique_labels": uniq_preview,
            "min_label": float(np.min(vals_np)),
            "max_label": float(np.max(vals_np)),
            "inferred_k": None,
            "notes": f"Targets not integer-like (max fractional part={max_frac:.3g}). Cannot infer K.",
        }

    vals_int = np.round(vals_np).astype(np.int64)
    uniq_int = np.unique(vals_int)
    min_label = int(uniq_int.min())
    max_label = int(uniq_int.max())

    if min_label < 0:
        inferred_k = None
        notes = "Found negative labels; cannot infer K via max+1."
    else:
        inferred_k = max_label + 1
        notes = "K inferred as max_label+1 from integer class IDs."

    uniq_preview = uniq_int[:50].tolist()
    if uniq_int.size > 50:
        uniq_preview.append("...")

    return {
        "n_labeled": n_labeled,
        "unique_labels": uniq_preview,
        "n_unique": int(uniq_int.size),
        "min_label": min_label,
        "max_label": max_label,
        "inferred_k": inferred_k,
        "notes": notes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("pt_file", type=str, help="Path to saved torch file with preds_by_layer and target")
    ap.add_argument("--write_json", type=str, default="", help="Optional path to write full report as JSON")
    args = ap.parse_args()

    data = torch.load(args.pt_file, map_location="cpu")
    if not isinstance(data, dict) or "target" not in data or "preds_by_layer" not in data:
        raise ValueError("Expected dict with keys {'preds_by_layer','target'}")

    preds_by_layer = data["preds_by_layer"]
    target = data["target"]

    if not isinstance(preds_by_layer, (list, tuple)) or len(preds_by_layer) == 0:
        raise ValueError("'preds_by_layer' must be a non-empty list/tuple of tensors")
    if not torch.is_tensor(target) or target.ndim != 2:
        raise ValueError("'target' must be a 2D tensor [N, P]")

    N, P = target.shape
    first_pred = preds_by_layer[0]
    if not torch.is_tensor(first_pred) or first_pred.ndim != 2:
        raise ValueError("Each preds_by_layer entry must be a 2D tensor [N, D]")

    D = int(first_pred.shape[1])

    print(f"Loaded: {args.pt_file}")
    print(f"target shape: (N={N}, P={P})")
    print(f"preds_by_layer: {len(preds_by_layer)} layers; first pred dim D={D}")

    per_prop = []
    inferred_ks = []
    for i in range(P):
        stats = infer_k_from_target_column(target[:, i])
        per_prop.append(stats)
        inferred_ks.append(stats["inferred_k"])

    # Replace None with 0 for sum check (unknowns break the sum)
    known_ks = [k for k in inferred_ks if isinstance(k, int)]
    unknown_count = sum(k is None for k in inferred_ks)

    print("\n=== Per-property inferred K (from target labels) ===")
    for i, st in enumerate(per_prop):
        print(
            f"prop {i:02d}: n_labeled={st['n_labeled']}, "
            f"min={st['min_label']}, max={st['max_label']}, "
            f"n_unique={st.get('n_unique', 'NA')}, inferred_k={st['inferred_k']}"
        )

    print("\n=== Cross-check ===")
    if unknown_count == 0:
        sum_k = int(sum(known_ks))
        print(f"sum(inferred K) = {sum_k}, pred width D = {D}")
        if sum_k != D:
            print("WARNING: sum(K) != D.")
            print("This usually means your model uses a different encoding than 'K = max_label+1'.")
            print("Common cases:")
            print("  - Binary properties encoded with 1 logit (sigmoid) while labels are {0,1} -> inferred K=2 but logits width uses 1")
            print("  - Some properties share heads / are packed differently than target columns")
        else:
            print("OK: sum(K) matches pred width.")
    else:
        print(f"Could not infer K for {unknown_count}/{P} properties (non-integer-like targets or unlabeled).")
        print(f"Known Ks sum to {sum(known_ks)} (ignoring unknowns); pred width D = {D}")

    # Optional: check all layers have same D
    bad = []
    for li, pr in enumerate(preds_by_layer):
        if not torch.is_tensor(pr) or pr.ndim != 2:
            bad.append((li, "not a 2D tensor"))
            continue
        if pr.shape[1] != D:
            bad.append((li, f"D={pr.shape[1]}"))
    if bad:
        print("\nWARNING: Not all layers share the same pred width as layer 0:")
        for li, why in bad[:20]:
            print(f"  layer {li}: {why}")
        if len(bad) > 20:
            print("  ...")

    report: Dict[str, Any] = {
        "pt_file": args.pt_file,
        "target_shape": [int(N), int(P)],
        "n_layers": int(len(preds_by_layer)),
        "pred_width_D": int(D),
        "per_property": per_prop,
        "inferred_K_list": inferred_ks,
    }

    if args.write_json:
        with open(args.write_json, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nWrote JSON report to: {args.write_json}")


if __name__ == "__main__":
    main()

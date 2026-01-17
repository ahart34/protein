#!/usr/bin/env python3
"""
inspect_pt.py

Inspect a Torch-saved .pt file produced by your task, e.g.:
  {"preds_by_layer": [Tensor, Tensor, ...], "target": Tensor}

This script is written to be compatible with older PyTorch versions that may NOT
have torch.nanmean / torch.nanmin / torch.nanmax.

Usage:
  python inspect_pt.py --pt YOUR_OUT_FILE.pt
  python inspect_pt.py --pt YOUR_OUT_FILE.pt --sample_rows 3 --max_layers 20
"""

import argparse
from typing import Any, List, Optional, Tuple

import torch


# ---------------------------
# Compatibility helpers
# ---------------------------

def has_nans(x: torch.Tensor) -> bool:
    return x.is_floating_point() and torch.isnan(x).any().item()


def nanmean_compat(x: torch.Tensor) -> torch.Tensor:
    """
    torch.nanmean compatibility for older PyTorch versions.
    Returns mean over non-NaN entries if NaNs exist, else mean().
    """
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    if has_nans(x):
        m = ~torch.isnan(x)
        if m.any().item():
            return x[m].mean()
        return torch.tensor(float("nan"))
    return x.mean()


def nanmin_compat(x: torch.Tensor) -> torch.Tensor:
    """
    torch.nanmin compatibility for older PyTorch versions.
    Returns min over non-NaN entries if NaNs exist, else min().
    """
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    if has_nans(x):
        m = ~torch.isnan(x)
        if m.any().item():
            return x[m].min()
        return torch.tensor(float("nan"))
    return x.min()


def nanmax_compat(x: torch.Tensor) -> torch.Tensor:
    """
    torch.nanmax compatibility for older PyTorch versions.
    Returns max over non-NaN entries if NaNs exist, else max().
    """
    if x.numel() == 0:
        return torch.tensor(float("nan"))
    if has_nans(x):
        m = ~torch.isnan(x)
        if m.any().item():
            return x[m].max()
        return torch.tensor(float("nan"))
    return x.max()


def safe_item(x: torch.Tensor) -> float:
    try:
        return x.item()
    except Exception:
        return float("nan")


# ---------------------------
# Inspection logic
# ---------------------------

def is_prob_like(x: torch.Tensor, atol: float = 1e-2) -> Tuple[bool, Optional[Tuple[float, float, float]]]:
    """
    Heuristic: values mostly in [0,1]. If 2D, also report row-sum stats.
    Note: multilabel sigmoid probabilities won't sum to 1; multiclass softmax will.
    Returns (is_prob, (row_sum_mean, row_sum_min, row_sum_max)) where tuple is None if not applicable.
    """
    if not isinstance(x, torch.Tensor) or x.numel() == 0:
        return False, None

    xmin = safe_item(x.min())
    xmax = safe_item(x.max())
    if xmin < 0 or xmax > 1:
        return False, None

    if x.ndim == 2:
        row_sums = x.sum(dim=-1)
        stats = (safe_item(row_sums.mean()), safe_item(row_sums.min()), safe_item(row_sums.max()))
        return True, stats

    return True, None


def print_tensor_summary(name: str, t: torch.Tensor) -> None:
    print(f"{name}:")
    print(f"  shape:  {tuple(t.shape)}")
    print(f"  dtype:  {t.dtype}")
    print(f"  device: {t.device}")
    t_float = t.float() if t.is_floating_point() else t
    tmin = safe_item(nanmin_compat(t_float))
    tmax = safe_item(nanmax_compat(t_float))
    tmean = safe_item(nanmean_compat(t_float))
    print(f"  min/max: {tmin:.6g} / {tmax:.6g}")
    print(f"  mean:   {tmean:.6g}")


def infer_task_structure(pred0: torch.Tensor, target: Optional[torch.Tensor]) -> str:
    """
    Rough inference: multiclass vs multilabel vs binary vs unknown.
    """
    if target is None or not isinstance(target, torch.Tensor):
        return "unknown (target is None or not a tensor)"

    if pred0.ndim == 2 and target.ndim == 1:
        return "multiclass (pred [N,C], target [N])"
    if pred0.ndim == 2 and target.ndim == 2:
        return "multilabel (pred [N,C], target [N,L])"
    if pred0.ndim == 1 and target.ndim == 1:
        return "binary (pred [N], target [N])"
    return f"unknown (pred ndim={pred0.ndim}, target ndim={target.ndim})"


def maybe_print_unique_target(target: torch.Tensor, max_show: int = 10) -> None:
    if target.numel() == 0:
        print("  target is empty")
        return

    if target.is_floating_point():
        mask = ~torch.isnan(target)
        vals = target[mask]
        nan_count = int(torch.isnan(target).sum().item())
        print("  NaN count:", nan_count)
    else:
        vals = target

    if vals.numel() == 0:
        print("  target has no labeled (non-NaN) entries")
        return

    # If float targets but meant as ints, try rounding for display
    if vals.is_floating_point():
        rounded = torch.round(vals)
        if safe_item((vals - rounded).abs().max()) < 1e-3:
            vals_disp = rounded.to(torch.int64)
        else:
            vals_disp = vals
    else:
        vals_disp = vals.to(torch.int64) if vals.dtype != torch.int64 else vals

    try:
        uniq = torch.unique(vals_disp)
        print(f"  unique labeled values: count={uniq.numel()}")
        if uniq.numel() > 0:
            show = uniq[:max_show].tolist()
            print(f"  first {min(max_show, uniq.numel())}: {show}")
    except Exception as e:
        print("  could not compute unique values:", e)


def detect_batch_vs_full(preds_by_layer: List[torch.Tensor], target: Optional[torch.Tensor]) -> None:
    print("\n=== Batch vs full-dataset check ===")
    n0s = []
    for p in preds_by_layer:
        if isinstance(p, torch.Tensor) and p.ndim >= 1:
            n0s.append(int(p.shape[0]))

    if not n0s:
        print("  Could not compute first-dimension sizes (no tensors?)")
        return

    print("  first-dim sizes (first 20):", n0s[:20])
    if len(set(n0s)) > 1:
        print("  → mixed sizes across layers → likely saved per-batch OR inconsistent saving")
    else:
        print("  → same size across layers → likely full-dataset per layer")

    if isinstance(target, torch.Tensor) and target.ndim >= 1:
        print("  target first-dim size:", int(target.shape[0]))


# ---------------------------
# Main
# ---------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pt", required=True, help="Path to the .pt file")
    ap.add_argument("--max_layers", type=int, default=20, help="How many layers to summarize (default: 20)")
    ap.add_argument("--sample_rows", type=int, default=3, help="How many rows/examples to print (default: 3)")
    ap.add_argument("--atol_row_sum", type=float, default=1e-2, help="Tolerance for row-sum≈1 heuristic")
    args = ap.parse_args()

    data: Any = torch.load(args.pt, map_location="cpu")

    print("=== Top-level structure ===")
    print("type:", type(data))
    if isinstance(data, dict):
        print("keys:", list(data.keys()))
    else:
        print("WARNING: top-level object is not a dict; inspection will be limited.")

    preds_by_layer = data.get("preds_by_layer") if isinstance(data, dict) else None
    target = data.get("target") if isinstance(data, dict) else None

    print("\n=== preds_by_layer ===")
    print("type:", type(preds_by_layer))
    if isinstance(preds_by_layer, list):
        print("length (# saved entries):", len(preds_by_layer))
    else:
        print("WARNING: preds_by_layer is not a list; got:", type(preds_by_layer))

    # Basic pred0 + target summaries
    if isinstance(preds_by_layer, list) and len(preds_by_layer) > 0 and isinstance(preds_by_layer[0], torch.Tensor):
        p0 = preds_by_layer[0]
        print_tensor_summary("preds_by_layer[0]", p0)

        prob_like, row_sum_stats = is_prob_like(p0, atol=args.atol_row_sum)
        if prob_like:
            print("  → values in [0,1] (probability-like)")
            if row_sum_stats is not None:
                rmean, rmin, rmax = row_sum_stats
                print(f"  row-sum stats (mean/min/max): {rmean:.6g} / {rmin:.6g} / {rmax:.6g}")
                if abs(rmean - 1.0) <= args.atol_row_sum:
                    print("  → row sums ~ 1 (multiclass softmax-like)")
                else:
                    print("  → row sums not ~ 1 (could be multilabel sigmoid probs)")
        else:
            print("  → not confined to [0,1] (logit-like)")

        print("\n=== target ===")
        if target is None:
            print("target is None")
        elif isinstance(target, torch.Tensor):
            print_tensor_summary("target", target)
            maybe_print_unique_target(target)
            print("\n=== Task structure inference ===")
            print(" ", infer_task_structure(p0, target))
        else:
            print("target exists but is not a torch.Tensor:", type(target))

    # Per-layer + sample rows
    if isinstance(preds_by_layer, list) and len(preds_by_layer) > 0 and all(isinstance(p, torch.Tensor) for p in preds_by_layer):
        detect_batch_vs_full(preds_by_layer, target)

        print("\n=== Per-layer summary (first max_layers) ===")
        for i, p in enumerate(preds_by_layer[: args.max_layers]):
            p_float = p.float() if p.is_floating_point() else p
            pmin = safe_item(nanmin_compat(p_float))
            pmax = safe_item(nanmax_compat(p_float))
            pmean = safe_item(nanmean_compat(p_float))
            print(
                f"  layer {i:02d} | shape {tuple(p.shape)} | "
                f"min {pmin:.3g} | max {pmax:.3g} | mean {pmean:.3g}"
            )

        print("\n=== Sample rows (layer 0) ===")
        p0 = preds_by_layer[0]
        if p0.ndim >= 1:
            n = min(args.sample_rows, int(p0.shape[0]))
        else:
            n = 0

        for i in range(n):
            print(f"\n  example {i}:")
            try:
                print("    pred:", p0[i])
            except Exception as e:
                print("    pred: <could not print>", e)

            if isinstance(target, torch.Tensor) and target.ndim >= 1 and i < target.shape[0]:
                try:
                    print("    target:", target[i])
                except Exception as e:
                    print("    target: <could not print>", e)

    print("\nDone.")


if __name__ == "__main__":
    main()

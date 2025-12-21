# early_exit_runner.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal

import time
import torch


Mode = Literal["early_exit", "normal", "analysis"]


# ----------------------------- utilities -----------------------------

def ascii_encode_2000(seqs: List[str], device: torch.device, max_len: int = 2000) -> torch.Tensor:
    # [B, <=2000] padded to 2000
    mats = []
    for s in seqs:
        arr = [ord(c) for c in s[:max_len]]
        if len(arr) < max_len:
            arr += [0] * (max_len - len(arr))
        mats.append(arr)
    return torch.tensor(mats, device=device, dtype=torch.int64)


@dataclass
class EncodedBatch:
    state: Dict[str, torch.Tensor]
    batch_size: int
    seq_ascii: torch.Tensor  # [B,2000]


@dataclass
class ExitPolicy:
    threshold: float
    select_last: bool = False


# ----------------------------- head adapters -----------------------------

class HeadAdapter:
    """
    Default: multi-label-ish confidence = max(sigmoid(logits)).
    For single-label softmax, use SoftmaxHeadAdapter below.
    """
    def logits(self, mlp, layer_idx: int, rep: torch.Tensor) -> torch.Tensor:
        return mlp[layer_idx](rep)

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(logits)

    def confidence(self, logits: torch.Tensor) -> torch.Tensor:
        p = self.probs(logits)
        return p.view(p.size(0), -1).max(dim=1).values  # [N]

    def pred_label(self, logits: torch.Tensor) -> torch.Tensor:
        p = self.probs(logits)
        return p.max(dim=1).indices  # [N]


class SoftmaxHeadAdapter(HeadAdapter):
    """Single-label classification: confidence = max(softmax(logits))."""
    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.softmax(logits, dim=-1)

    def pred_label(self, logits: torch.Tensor) -> torch.Tensor:
        return torch.argmax(logits, dim=-1)


# ----------------------------- model adapters -----------------------------

class ModelAdapter:
    """Model-specific tokenize/init + one-layer forward + head rep."""
    def num_layers(self) -> int:
        raise NotImplementedError

    def encode(self, sequences: List[str], device: torch.device) -> EncodedBatch:
        raise NotImplementedError

    def forward_one_layer(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> None:
        """In-place update enc.state for active samples."""
        raise NotImplementedError

    def get_layer_rep_for_head(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> torch.Tensor:
        """Return pooled representation [N_active, H]."""
        raise NotImplementedError


class ESMAdapter(ModelAdapter):
    """
    Expects wrapper like your TorchDrug ESM task uses:
      model.alphabet.get_batch_converter()
      model.model.layers, model.model.embed_tokens, model.model.padding_idx, model.model.emb_layer_norm_after
    """
    def __init__(self, esm_wrapper):
        self.m = esm_wrapper

    def num_layers(self) -> int:
        return int(self.m.model.num_layers)

    def encode(self, sequences: List[str], device: torch.device) -> EncodedBatch:
        inp = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, toks = self.m.alphabet.get_batch_converter()(inp)
        toks = toks.to(device)
        padding = toks.eq(self.m.model.padding_idx)

        x = self.m.model.embed_scale * self.m.model.embed_tokens(toks)
        if padding.any():
            x = x * (1 - padding.unsqueeze(-1).type_as(x))
        x = x.transpose(0, 1)  # [T,B,E]

        return EncodedBatch(
            state={"x": x, "padding": padding},
            batch_size=len(sequences),
            seq_ascii=ascii_encode_2000(sequences, device),
        )

    def forward_one_layer(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> None:
        x = enc.state["x"]
        padding = enc.state["padding"]

        hs = x[:, active, :]
        out = self.m.model.layers[layer_idx](
            hs,
            self_attn_padding_mask=padding[active] if padding is not None else None,
            need_head_weights=False,
        )
        hs = out[0] if isinstance(out, tuple) else out
        x[:, active, :] = hs

        # ESM2 final LN after last layer
        if layer_idx == self.m.model.num_layers - 1:
            x[:, active, :] = self.m.model.emb_layer_norm_after(x[:, active, :])

        enc.state["x"] = x

    def get_layer_rep_for_head(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> torch.Tensor:
        x = enc.state["x"][:, active, :]      # [T,N,E]
        return x.transpose(0, 1).mean(dim=1)  # [N,E]


class ProtBERTAdapter(ModelAdapter):
    """
    Expects HuggingFace BERT-ish model:
      model.embeddings
      model.encoder.layer
      model.config.num_hidden_layers
    Tokenizer must already be constructed outside.
    """
    def __init__(self, bert_model, tokenizer, max_length: int = 1024):
        self.m = bert_model
        self.tok = tokenizer
        self.max_length = max_length

    def num_layers(self) -> int:
        return int(self.m.config.num_hidden_layers)

    @staticmethod
    def _prep(seqs: List[str]) -> List[str]:
        # ProtBERT convention: space-separated, U/O->X
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def encode(self, sequences: List[str], device: torch.device) -> EncodedBatch:
        enc = self.tok(
            self._prep(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0  # [B,1,1,L]

        hs = self.m.embeddings(input_ids)  # [B,L,H]

        return EncodedBatch(
            state={"hs": hs, "ext_mask": ext_mask},
            batch_size=len(sequences),
            seq_ascii=ascii_encode_2000(sequences, device),
        )

    def forward_one_layer(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> None:
        hs = enc.state["hs"]
        ext = enc.state["ext_mask"]

        hs_a = self.m.encoder.layer[layer_idx](
            hs[active],
            attention_mask=ext[active],
            head_mask=None,
            output_attentions=False,
        )[0]
        hs[active] = hs_a
        enc.state["hs"] = hs

    def get_layer_rep_for_head(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> torch.Tensor:
        hs = enc.state["hs"][active]  # [N,L,H]
        return hs.mean(dim=1)         # [N,H]


class ProtAlbertAdapter(ModelAdapter):
    """
    Expects HuggingFace ALBERT-ish model:
      model.embeddings
      model.encoder.embedding_hidden_mapping_in
      model.encoder.albert_layer_groups
      model.config.num_hidden_layers / num_hidden_groups
    Tokenizer must already be constructed outside.
    """
    def __init__(self, albert_model, tokenizer, max_length: int = 550):
        self.m = albert_model
        self.tok = tokenizer
        self.max_length = max_length

        self.n_layers = int(self.m.config.num_hidden_layers)
        self.n_groups = int(self.m.config.num_hidden_groups)
        self.layers_per_group = self.n_layers // self.n_groups

    def num_layers(self) -> int:
        return self.n_layers

    @staticmethod
    def _prep(seqs: List[str]) -> List[str]:
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def encode(self, sequences: List[str], device: torch.device) -> EncodedBatch:
        enc = self.tok(
            self._prep(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        ).to(device)

        input_ids = enc["input_ids"]
        attn_mask = enc["attention_mask"]
        attn_ext = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        hs = self.m.embeddings(input_ids)                    # [B,L,E_word]
        hs = self.m.encoder.embedding_hidden_mapping_in(hs)  # [B,L,H]

        return EncodedBatch(
            state={"hs": hs, "attn_ext": attn_ext},
            batch_size=len(sequences),
            seq_ascii=ascii_encode_2000(sequences, device),
        )

    def forward_one_layer(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> None:
        hs = enc.state["hs"]
        attn_ext = enc.state["attn_ext"]

        group_idx = layer_idx // self.layers_per_group
        grp = self.m.encoder.albert_layer_groups[group_idx]

        hs_a = grp(
            hs[active],
            attention_mask=attn_ext[active],
            head_mask=[None] * self.n_layers,
            output_attentions=False,
        )[0]
        hs[active] = hs_a
        enc.state["hs"] = hs

    def get_layer_rep_for_head(self, enc: EncodedBatch, layer_idx: int, active: torch.Tensor) -> torch.Tensor:
        hs = enc.state["hs"][active]
        return hs.mean(dim=1)


# ----------------------------- runner -----------------------------

class EarlyExitRunner:
    def __init__(self, adapter: ModelAdapter, head: Optional[HeadAdapter] = None):
        self.adapter = adapter
        self.head = head or HeadAdapter()

# inside early_exit_runner.py  (replace EarlyExitRunner.run)

    @torch.no_grad()
    def run(
        self,
        sequences: List[str],
        mlp,
        device: torch.device,
        mode: Mode,
        policy: ExitPolicy,
        *,
        store_layerwise: bool = False,
        store_prob_full: bool = False,
    ) -> Dict[str, torch.Tensor]:

        enc = self.adapter.encode(sequences, device)
        B = enc.batch_size
        L = self.adapter.num_layers()

        # GPU-accurate walltime
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        computed_layers = torch.full((B,), -1, device=device, dtype=torch.int64)

        # -------- NORMAL MODE: compute head at every layer (per-layer logits) --------
        if mode == "normal":
            active_all = torch.arange(B, device=device)

            # We will store pooled reps per layer: [B, L, H]
            reps_per_layer = None

            # backbone cumulative timing after each layer (NO head timing inside loop)
            backbone_cum_ms = torch.empty((L,), device=device, dtype=torch.float32)

            # optional analysis tensors (computed from head probs later)
            layer_conf = None
            layer_label = None
            if store_layerwise:
                layer_conf = torch.full((B, L), float("nan"), device=device)
                layer_label = torch.full((B, L), -1, device=device, dtype=torch.int64)

            # optional full prob cube (computed later)
            layer_prob_full = None

            # ---- 1) Backbone pass only: store reps + time per layer ----
            for l in range(L):
                computed_layers[:] = l

                self.adapter.forward_one_layer(enc, l, active_all)

                # pooled rep for this layer (no head yet)
                rep = self.adapter.get_layer_rep_for_head(enc, l, active_all)  # [B,H]
                if reps_per_layer is None:
                    H = rep.size(1)
                    reps_per_layer = torch.empty((B, L, H), device=device, dtype=rep.dtype)
                reps_per_layer[:, l, :] = rep

                # cumulative backbone time up to layer l
                if device.type == "cuda":
                    torch.cuda.synchronize()
                backbone_cum_ms[l] = (time.perf_counter() - t0) * 1000.0

            # ---- 2) Head-only pass: compute logits per layer + time heads independently ----
            head_ms = torch.empty((L,), device=device, dtype=torch.float32)
            pred_layers = None  # [B,L,C]

            for l in range(L):
                if device.type == "cuda":
                    torch.cuda.synchronize()
                h0 = time.perf_counter()

                logits = self.head.logits(mlp, l, reps_per_layer[:, l, :])  # [B,C]

                if device.type == "cuda":
                    torch.cuda.synchronize()
                head_ms[l] = (time.perf_counter() - h0) * 1000.0

                if pred_layers is None:
                    C = logits.size(1)
                    pred_layers = torch.empty((B, L, C), device=device, dtype=logits.dtype)
                pred_layers[:, l, :] = logits

                if store_layerwise or store_prob_full:
                    probs = self.head.probs(logits)  # [B,C]
                    if store_layerwise:
                        layer_conf[:, l] = probs.view(B, -1).max(dim=1).values
                        layer_label[:, l] = torch.argmax(probs, dim=-1)
                    if store_prob_full:
                        if layer_prob_full is None:
                            layer_prob_full = torch.full((B, L, C), float("nan"),
                                                         device=device, dtype=probs.dtype)
                        layer_prob_full[:, l, :] = probs

            # ---- combine into the “normal runtime per layer” you want ----
            normal_layer_time_ms = backbone_cum_ms + head_ms  # [L]
            wall_ms = normal_layer_time_ms[-1]  # time to reach final layer + head(final)

            out: Dict[str, torch.Tensor] = {
                "sequences": enc.seq_ascii,
                "pred_layers": pred_layers,                 # [B,L,C] logits at each layer
                "computed_layers": computed_layers,         # [B] == L-1
                "backbone_cum_ms": backbone_cum_ms,         # [L] (no head time included)
                "head_ms": head_ms,                         # [L] (head-only time)
                "layer_walltime_ms": normal_layer_time_ms,  # [L] = backbone_cum + head(l)
                "walltime_ms": wall_ms,                     # scalar = layer_walltime_ms[-1]
            }
            if store_layerwise:
                out.update({"layer_conf": layer_conf, "layer_label": layer_label})
            if store_prob_full and layer_prob_full is not None:
                out["layer_prob_full"] = layer_prob_full

            return out

        # -------- EARLY_EXIT / ANALYSIS: keep existing behavior --------
        active = torch.arange(B, device=device)
        final_layer = torch.full((B,), -1, device=device, dtype=torch.int64)
        final_logits: List[Optional[torch.Tensor]] = [None] * B

        best_conf = torch.full((B,), -float("inf"), device=device)
        best_layer = torch.full((B,), -1, device=device, dtype=torch.int64)
        best_logits: List[Optional[torch.Tensor]] = [None] * B

        layer_conf = None
        layer_label = None
        layer_prob_full = None
        if store_layerwise:
            layer_conf = torch.full((B, L), float("nan"), device=device)
            layer_label = torch.full((B, L), -1, device=device, dtype=torch.int64)

        for l in range(L):
            if active.numel() == 0:
                break

            computed_layers[active] = l

            self.adapter.forward_one_layer(enc, l, active)
            rep = self.adapter.get_layer_rep_for_head(enc, l, active)
            logits = self.head.logits(mlp, l, rep)
            probs = self.head.probs(logits)
            conf = probs.view(probs.size(0), -1).max(dim=1).values

            if store_layerwise:
                layer_conf[active, l] = conf
                layer_label[active, l] = torch.argmax(probs, dim=-1)

            if store_prob_full:
                if layer_prob_full is None:
                    C = probs.size(1)
                    layer_prob_full = torch.full((B, L, C), float("nan"), device=device, dtype=probs.dtype)
                layer_prob_full[active, l, :] = probs

            # update best-so-far
            is_final = (l == L - 1)
            better = conf > best_conf[active]
            if is_final and policy.select_last:
                better = torch.ones_like(better, dtype=torch.bool)
            if better.any():
                gi = active[better]
                best_conf[gi] = conf[better]
                best_layer[gi] = l
                logits_b = logits[better]
                for j, g in enumerate(gi.tolist()):
                    best_logits[g] = logits_b[j]

            if mode == "analysis":
                continue

            # early-exit decision
            exit_mask = conf > policy.threshold
            newly = active[exit_mask]
            still = active[~exit_mask]
            if newly.numel() > 0:
                logits_new = logits[exit_mask]
                for j, g in enumerate(newly.tolist()):
                    final_logits[g] = logits_new[j]
                final_layer[newly] = l
            active = still

        # walltime
        if device.type == "cuda":
            torch.cuda.synchronize()
        wall_ms = torch.tensor((time.perf_counter() - t0) * 1000.0, device=device)

        out: Dict[str, torch.Tensor] = {
            "sequences": enc.seq_ascii,
            "computed_layers": computed_layers,
            "walltime_ms": wall_ms,
        }

        if mode == "early_exit":
            for g in range(B):
                if final_logits[g] is None:
                    final_logits[g] = best_logits[g]
                    final_layer[g] = best_layer[g]
            out["pred"] = torch.stack(final_logits, dim=0)
            out["layers"] = final_layer

        if store_layerwise:
            out.update({"layer_conf": layer_conf, "layer_label": layer_label})
        if store_prob_full and layer_prob_full is not None:
            out["layer_prob_full"] = layer_prob_full

        return out

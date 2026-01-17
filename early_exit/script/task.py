from collections.abc import Sequence

import time

import torch
from torch import nn
from torch.nn import functional as F

from torch_scatter import scatter_add

from torchdrug import core, layers, tasks, metrics, data
from torchdrug.core import Registry as R
import pickle
import os
import csv
from collections import defaultdict
from torchdrug.layers import functional
import math
import numpy as np
from transformers import AutoTokenizer, AlbertModel
import torch.nn as nn

def graphs_to_sequences(graphs, data):
    return [
        "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
        for g in graphs
    ]


class MLP(nn.Module):
    def __init__(self,
                 in_channels: int,
                 mid_channels: int,
                 out_channels: int,
                 batch_norm: bool,
                 dropout: float = 0.0,
                 bias: bool = True,
                 leakyrelu_negative_slope: float = 0.2,
                 momentum: float = 0.2) -> nn.Module:
        super(MLP, self).__init__()

        module = []
        if batch_norm:
            module.append(nn.BatchNorm1d(in_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        module.append(nn.Dropout(dropout))
        if mid_channels is None:
            module.append(nn.Linear(in_channels, out_channels, bias = bias))
        else:
            module.append(nn.Linear(in_channels, mid_channels, bias = bias))
        if batch_norm:
            if mid_channels is None:
                module.append(nn.BatchNorm1d(out_channels, momentum=momentum))
            else:
                module.append(nn.BatchNorm1d(mid_channels, momentum=momentum))
        module.append(nn.LeakyReLU(leakyrelu_negative_slope))
        if mid_channels is None:
            module.append(nn.Dropout(dropout))
        else:
            module.append(nn.Linear(mid_channels, out_channels, bias = bias))

        self.module = nn.Sequential(*module)

    def forward(self, input):
        return self.module(input)
    

@R.register("tasks.FunctionAnnotation_AllLayers")
class FunctionAnnotation_AllLayers(tasks.Task, core.Configurable):

    def __init__(self, model, num_class=1, metric=('auprc@micro', 'f1_max'), weight=None, graph_construction_model=None, 
                 mlp_batch_norm=False, attn_head_mask = None, mlp_dropout=0, verbose=0):
        super(FunctionAnnotation_AllLayers, self).__init__()
        self.model = model

        # Set up MLPs for each layer's output
        num_layers = model.num_layers
        model.attn_head_mask = attn_head_mask
        #num_layers = 33 
        self.mlp = nn.ModuleList([
            MLP(in_channels=model.output_dim, mid_channels=model.output_dim, 
                out_channels=num_class, batch_norm=mlp_batch_norm, dropout=mlp_dropout) 
            for _ in range(num_layers)
        ])
        for i, mlp in enumerate(self.mlp[:3]):            # show the first 3 layers for brevity
            last = mlp.module[-1]                         # the final Linear
            print(f"[INIT] layer {i} Linear({last.in_features}, {last.out_features})")
        print(f"[INIT] total MLP layers: {len(self.mlp)}")

        if weight is None:
            weight = torch.ones((num_class,), dtype=torch.float)
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.metric = metric
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

    def forward(self, batch):
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        # Run prediction once and get predictions for all layers

        preds = self.predict(batch, all_loss, metric)

        target = self.target(batch)

        # Calculate and accumulate loss for each layer
        loss_fn = torch.nn.BCELoss(weight=torch.as_tensor(self.weight))

        for pred in preds:
            loss = loss_fn(pred.sigmoid(), target)
            name = tasks._get_criterion_name("bce")
            metric[name] = loss
            all_loss += loss

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graphs = batch["graph"]
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        representations = self.model(sequences, repr_layers=list(range(1, self.model.num_layers + 1)))
        preds = []
        for layer_idx, layer_mlp in enumerate(self.mlp): ###ISSUE OCCURRING HERE!!!!
            layer_output = representations[layer_idx+1]
            preds.append(layer_mlp(layer_output))
        return preds

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        metric = {}
        total_score_f1 = 0
        for layer_idx, pred in enumerate(preds):
            for _metric in self.metric:
                if _metric == "auroc@micro":
                    score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
                elif _metric == "auprc@micro":
                    score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
                elif _metric == "f1_max":
                    score = metrics.f1_max(pred, target)
                    total_score_f1 += score
                else:
                    raise ValueError("Unknown criterion `%s`" % _metric)
                
                name = tasks._get_metric_name(_metric)
                metric[f"{name} Layer {layer_idx}"] = score
        metric["f1_max"] = total_score_f1
        if os.getenv("CONFIDENCE_CALIBRATION"): 
            metric["preds"] = [p.detach().cpu() for p in preds]
            metric["target"] = target.detach().cpu()
        return metric   

@R.register("tasks.PropertyPredictionAllLayers")
class PropertyPredictionAllLayers(tasks.Task, core.Configurable):
    """
    Graph / molecule / protein property prediction task.

    This class is also compatible with semi-supervised learning.

    Parameters:
        model (nn.Module): graph representation model
        task (str, list or dict, optional): training task(s).
            For dict, the keys are tasks and the values are the corresponding weights.
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions and the values
            are the corresponding weights. Available criterions are ``mse``, ``bce`` and ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc`` and ``auroc``.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target
        num_class (int, optional): number of classes
        mlp_batch_norm (bool, optional): apply batch normalization in mlp or not
        mlp_dropout (float, optional): dropout in mlp
        graph_construction_model (nn.Module, optional): graph construction model
        verbose (int, optional): output verbose level
    """

    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=1,
                 normalization=True, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, verbose=0):
        super(PropertyPredictionAllLayers, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute the mean and derivation for each task on the training set.
        """
        values = defaultdict(list)
        for sample in train_set:
            if not sample.get("labeled", True):
                continue
            for task in self.task:
                if not math.isnan(sample[task]):
                    values[task].append(sample[task])
        mean = []
        std = []
        weight = []
        num_class = []
        for task, w in self.task.items():
            value = torch.tensor(values[task])
            mean.append(value.float().mean())
            std.append(value.float().std())
            weight.append(w)
            if value.ndim > 1:
                num_class.append(value.shape[1])
            elif value.dtype == torch.long:
                task_class = value.max().item()
                if task_class == 1 and "bce" in self.criterion:
                    num_class.append(1)
                else:
                    num_class.append(task_class + 1)
            else:
                num_class.append(1)

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.register_buffer("weight", torch.as_tensor(weight, dtype=torch.float))
        self.num_class = self.num_class or num_class
        print(f"self.num_class {self.num_class}")

        hidden_dims = [self.model.output_dim] * (self.num_mlp_layer - 1)
        self.mlp = nn.ModuleList([layers.MLP(self.model.output_dim, hidden_dims + [sum(self.num_class)],
                            batch_norm=self.mlp_batch_norm, dropout=self.mlp_dropout)
                            for _ in range(self.model.num_layers) ])


    def forward(self, batch):
        """"""
        all_loss = torch.tensor(0, dtype=torch.float32, device=self.device)
        metric = {}

        preds = self.predict(batch, all_loss, metric)

        if all([t not in batch for t in self.task]):
            # unlabeled data
            return all_loss, metric

        target = self.target(batch)
        labeled = ~torch.isnan(target)
        target[~labeled] = 0



        for layer_idx, pred in enumerate(preds):
            #print(f"pred shape {pred.shape}")
            #print(f"target shape {target.shape}")
            for criterion, weight in self.criterion.items():
                if criterion == "mse":
                    if self.normalization:
                        loss = F.mse_loss((pred - self.mean) / self.std, (target - self.mean) / self.std, reduction="none")
                    else:
                        loss = F.mse_loss(pred, target, reduction="none")
                elif criterion == "bce":
                    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
                elif criterion == "ce":
                    loss = F.cross_entropy(pred, target.long().squeeze(-1), reduction="none").unsqueeze(-1)
                else:
                    raise ValueError("Unknown criterion `%s`" % criterion)
                loss = functional.masked_mean(loss, labeled, dim=0) #--> this was here before but not during running ppi

                name = tasks._get_criterion_name(criterion)
                if self.verbose > 0:
                    for t, l in zip(self.task, loss):
                        metric["Layer %d %s [%s]" % (layer_idx, name, t)] = l
                loss = (loss * self.weight).sum() / self.weight.sum()
                metric[name] = loss
                all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        graphs = batch["graph"]
        sequences = []
        device = next(self.model.parameters()).device

        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        #scaler = amp.GradScaler()
       #with amp.autocast():
        representations = self.model(sequences, repr_layers=list(range(1, self.model.num_layers + 1)))
        preds = []
        for layer_idx, layer_mlp in enumerate(self.mlp):
            layer_output = representations[layer_idx + 1]
            pred = layer_mlp(layer_output)
            if self.normalization:
                pred = pred * self.std + self.mean
            preds.append(pred)
        return preds

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target


    def evaluate(self, preds, target):
        labeled = ~torch.isnan(target)
        metric = {}
        total_score_acc = 0
        total_score_spearman = 0
        for layer_idx, pred in enumerate(preds):
            for _metric in self.metric:
                if _metric == "mae":
                    score = F.l1_loss(pred, target, reduction="none")
                    score = functional.masked_mean(score, labeled, dim=0)
                elif _metric == "rmse":
                    score = F.mse_loss(pred, target, reduction="none")
                    score = functional.masked_mean(score, labeled, dim=0).sqrt()
                elif _metric == "acc":
                    score = []
                    num_class = 0
                    for i, cur_num_class in enumerate(self.num_class):
                        _pred = pred[:, num_class:num_class + cur_num_class]
                        _target = target[:, i]
                        _labeled = labeled[:, i]
                        _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                        score.append(_score)
                        num_class += cur_num_class
                    score = torch.stack(score)
                    total_score_acc += score
                elif _metric == "mcc":
                    score = []
                    num_class = 0
                    for i, cur_num_class in enumerate(self.num_class):
                        _pred = pred[:, num_class:num_class + cur_num_class]
                        _target = target[:, i]
                        _labeled = labeled[:, i]
                        _score = metrics.matthews_corrcoef(_pred[_labeled], _target[_labeled].long())
                        score.append(_score)
                        num_class += cur_num_class
                    score = torch.stack(score)
                elif _metric == "auroc":
                    score = []
                    for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                        _score = metrics.area_under_roc(_pred[_labeled], _target[_labeled])
                        score.append(_score)
                    score = torch.stack(score)
                elif _metric == "auprc":
                    score = []
                    for _pred, _target, _labeled in zip(pred.t(), target.long().t(), labeled.t()):
                        _score = metrics.area_under_prc(_pred[_labeled], _target[_labeled])
                        score.append(_score)
                    score = torch.stack(score)
                elif _metric == "r2":
                    score = []
                    for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                        _score = metrics.r2(_pred[_labeled], _target[_labeled])
                        score.append(_score)
                    score = torch.stack(score)
                elif _metric == "spearmanr":
                    score = []
                    for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                        _score = metrics.spearmanr(_pred[_labeled], _target[_labeled])
                        score.append(_score)
                    score = torch.stack(score)
                    total_score_spearman += score
                elif _metric == "pearsonr":
                    score = []
                    for _pred, _target, _labeled in zip(pred.t(), target.t(), labeled.t()):
                        _score = metrics.pearsonr(_pred[_labeled], _target[_labeled])
                        score.append(_score)
                    score = torch.stack(score)
                else:
                    raise ValueError("Unknown metric `%s`" % _metric)

                name = tasks._get_metric_name(_metric)
                for t, s in zip(self.task, score):
                    metric[f"{name} Layer {layer_idx} [{t}]"] = s
                #metric["total_score_rmse"] = total_score_rmse
        metric["accuracy"] = total_score_acc

        return metric
    
@R.register("tasks.NodePropertyPredictionAllLayers")
class NodePropertyPredictionAllLayers(tasks.Task, core.Configurable):
    """
    Node / atom / residue property prediction task using representations from all layers.

    Parameters:
        model (nn.Module): graph representation model
        criterion (str, list or dict, optional): training criterion(s). For dict, the keys are criterions 
            and the values are the corresponding weights. Available criterions are ``mse``, ``bce``, ``ce``.
        metric (str or list of str, optional): metric(s).
            Available metrics are ``mae``, ``rmse``, ``auprc``, ``auroc``, etc.
        num_mlp_layer (int, optional): number of layers in mlp prediction head
        normalization (bool, optional): whether to normalize the target (for regression)
        num_class (int, optional): number of classes (for classification)
        verbose (int, optional): output verbose level
    """

    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(NodePropertyPredictionAllLayers, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

        # Prepare a list of MLP prediction heads, one per layer in the model
        if hasattr(self.model, "node_output_dim"):
            output_dim = self.model.node_output_dim
        else:
            output_dim = self.model.output_dim

        hidden_dims = [output_dim] * (self.num_mlp_layer - 1)
        # We assume the model has 'num_layers' to let us build heads for each
        self.mlp = nn.ModuleList([
            layers.MLP(
                input_dim=output_dim,
                hidden_dims=hidden_dims + [self.num_class],
            )
            for _ in range(self.model.num_layers)
        ])

    def forward(self, batch):
        """
        Sum the losses from each layer's prediction head.
        """
        all_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        metric = {}

        # Get a list of predictions, one for each layer
        preds = self.predict(batch)
        target_dict = self.target(batch)
        target_vals = target_dict["label"]
        mask = target_dict["mask"]
        labeled = ~torch.isnan(target_vals) & mask

        #print(f"preds {len(preds)} {preds[0].shape}")
        #print(f"target_vals {target_vals.shape}")

        # For each layer, compqute the loss
        for layer_idx, pred in enumerate(preds):
            for criterion, weight in self.criterion.items():
                if criterion == "mse":
                    if self.normalization:
                        loss = F.mse_loss(
                            (pred - self.mean) / self.std,
                            (target_vals - self.mean) / self.std,
                            reduction="none"
                        )
                    else:
                        loss = F.mse_loss(pred, target_vals, reduction="none")
                elif criterion == "bce":
                    loss = F.binary_cross_entropy_with_logits(
                        pred, target_vals.float(), reduction="none"
                    )
                elif criterion == "ce":
                    # For CE, we interpret target_vals as Long
                    try:
                        loss = F.cross_entropy(pred, target_vals.long(), reduction="none")
                    except ValueError as e: 
                        print(f"pred_shape {pred.shape} target_vals_sahpe {target_vals.shape}")

                else:
                    raise ValueError(f"Unknown criterion `{criterion}`")

                # Mask out invalid/NaN labels
                loss = functional.masked_mean(loss, labeled, dim=0)

                name = tasks._get_criterion_name(criterion)
                # To keep layer-specific logs, we add the layer index to the name
                metric[f"Layer {layer_idx} {name}"] = loss
                all_loss += loss * weight

        return all_loss, metric

    def predict(self, batch, all_loss=None, metric=None):
        """
        Return a list of predictions (one per layer).
        Each layer's node_feature is passed to a separate MLP.
        """

        graphs = batch["graph"]
        sequences = []
        device = next(self.model.parameters()).device

        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)

        representations = self.model(sequences, repr_layers=list(range(1, self.model.num_layers + 1)), per_residue=True)



        preds = []
        for layer_idx, layer_mlp in enumerate(self.mlp):
            layer_repr = representations[layer_idx + 1]
            seq_unpadded = []
            start_idx = 0
            for i, sequence in enumerate(sequences):
                seq_len = len(sequence)
                seq_unpadded.append(layer_repr[i, :seq_len, :])
            layer_repr_unpadded = torch.cat(seq_unpadded, dim=0)
            # forward through the MLP for this layer
            #print(f"layer_repr_unpadded {layer_repr_unpadded.shape}")
            layer_out = layer_mlp(layer_repr_unpadded)
            #print(f"layer_out {layer_out.shape}")
            if self.normalization:
                layer_out = layer_out * self.std + self.mean    
            preds.append(layer_out)
        return preds

    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }


    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])

        # Evaluate each layer
        for layer_idx, pred in enumerate(preds):
            #print(f"preds shape {preds.shape}")
            #print(f"target_shape {_target.shape}")
            for _metric in self.metric:
                if _metric in ["mae", "rmse"]:
                    # Typically for regression
                    if _metric == "mae":
                        score = F.l1_loss(pred, _target, reduction="none")
                    else:  # rmse
                        score = F.mse_loss(pred, _target, reduction="none").sqrt()

                    score = functional.masked_mean(score, labeled, dim=0)

                elif _metric in ["micro_auroc", "micro_auprc"]:
                    # Single "micro" approach across all labeled nodes
                    if _metric == "micro_auroc":
                        score = metrics.area_under_roc(pred[labeled], _target[labeled])
                    else:
                        score = metrics.area_under_prc(pred[labeled], _target[labeled])

                elif _metric in ["macro_auroc", "macro_auprc"]:
                    # "macro" means compute per-graph, then average
                    if _metric == "macro_auroc":
                        score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                    else:
                        score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

                elif _metric == "macro_acc":
                    # One typical approach for multi-class:
                    # (pred[labeled].argmax(-1) == _target[labeled]).float()
                    # Then do per-graph average
                    pred_argmax = pred[labeled].argmax(dim=-1)
                    correct = (pred_argmax == _target[labeled]).float()
                    score = functional.variadic_mean(correct, _size).mean()

                else:
                    raise ValueError(f"Unknown metric `{_metric}`")

                name = tasks._get_metric_name(_metric)
                metric[f"{name} Layer {layer_idx}"] = score

        return metric


@R.register("tasks.Classification_walltime_ProtBert")
class Classification_walltime_ProtBert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer. 
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(Classification_walltime_ProtBert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    # ------------- helper to space‑separate sequences ------------
    @staticmethod
    def _prep_protbert(seqs):               # EDIT: name
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    # --------------------------- PREDICT -------------------------
    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()

        graphs = batch["graph"]

        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        batch_size = len(sequences)

        # ---- tokenise ----
        prepared = self._prep_protbert(sequences)
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,                # ProtBERT was trained up to 1024
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---- temperatures / bookkeeping ----
        n_layers = self.model.config.num_hidden_layers
        layer_stop = int(os.getenv("LAYER"))

        # ---- build extended attention mask like BERT does ----
        ext_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # ---- initial embeddings ----
        hs = self.model.embeddings(input_ids)

        # ---- iterate through encoder layers (no weight sharing) ----
        for layer_idx, layer in enumerate(self.model.encoder.layer):     # EDIT: simple loop
            # BERT layer forward
            out = layer(
                hs,
                attention_mask=ext_mask,
                head_mask=None,
                output_attentions=False
            )
            hs = out[0]                 # first element is hidden states

            if layer_idx == layer_stop:
            # ---------- classifier ----------
                pooled   = hs.mean(dim=1)
                final_logits   = self.mlp[layer_idx](pooled)
                break

        return {
            "pred": final_logits,
        }

    # ------------- unchanged target / evaluate -----------------
    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)
        labeled = ~torch.isnan(target)
        metric_out = {}
        if isinstance(self.metric, dict):
            m = list(self.metric.keys())
        else:
            m = self.metric

        if m == "auroc@micro":
            score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
        elif m == "auprc@micro":
            score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
        elif m == "f1_max":
            score = metrics.f1_max(pred, target)
        elif m == "acc" or "acc" in m:
            m = "acc"
            score = []
            num_class = 0
            for i, cur_num_class in enumerate(self.num_class):
                _pred = pred[:, num_class:num_class + cur_num_class]
                _target = target[:, i]
                _labeled = labeled[:, i]
                _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                score.append(_score)
                num_class += cur_num_class
            score = torch.stack(score)
        else:
            raise ValueError(f"Unknown metric {m}")
        metric_out[m] = score.item()
        return metric_out

@R.register("tasks.Property_walltime_ProtBert")
class Property_walltime_ProtBert(Classification_walltime_ProtBert):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), metric=("acc"), criterion="mse", num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super().__init__(model=model, metric=metric, verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer)
        self.model = model
        self.task = task
        self.criterion = criterion
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False

    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target
    
@R.register("tasks.Property_Node_walltime_ProtBert")
class Property_Node_walltime_ProtBert(tasks.Task, core.Configurable):
    _option_members = {"criterion"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_acc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(Property_Node_walltime_ProtBert, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Forward the batch through ProtBERT up to `last_layer`
        (taken from the env-var LAYER).  Return per-residue logits
        from that layer only.

        Returns
        -------
        dict
            {"pred": final_logits}
        """
        # -----------------------------------------------------------
        # 0) set-up
        # -----------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        # -----------------------------------------------------------
        # 1) graphs → raw sequences
        # -----------------------------------------------------------
        graphs = batch["graph"]
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]

        # -----------------------------------------------------------
        # 2) tokenise
        # -----------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert",
                do_lower_case=False,
                use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )

        enc = self._tokenizer(
            self._prep_protbert(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # -----------------------------------------------------------
        # 3) embeddings + bert-style mask
        # -----------------------------------------------------------
        hs = self.model.embeddings(input_ids)                 # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # -----------------------------------------------------------
        # 4) figure out which layer to stop at
        # -----------------------------------------------------------
        n_layers = self.model.config.num_hidden_layers
        env_val   = os.getenv("LAYER", str(n_layers - 1))     # default = last layer
        last_layer = int(float(env_val))                      # allow "3" or "3.0"

        if not (0 <= last_layer < n_layers):
            raise ValueError(f"LAYER={last_layer} out of range (0-{n_layers-1})")

        # -----------------------------------------------------------
        # 5) forward only up to that layer
        # -----------------------------------------------------------
        with torch.no_grad():
            for lidx, layer in enumerate(self.model.encoder.layer):
                hs = layer(
                    hs,
                    attention_mask=ext_mask,
                    head_mask=None,
                    output_attentions=False,
                )[0]
                if lidx == last_layer:
                    break

        # -----------------------------------------------------------
        # 6) per-sample logits via the *same* layer’s MLP head
        # -----------------------------------------------------------
        final_logits = []
        for b, seq in enumerate(sequences):
            seq_len = len(seq)
            h_i   = hs[b, 1 : seq_len + 1, :]      # drop [CLS]
            log_i = self.mlp[last_layer](h_i)      # (L_res,C)
            final_logits.append(log_i)

        # -----------------------------------------------------------
        # 7) return only what you asked for
        # -----------------------------------------------------------
        return {"pred": final_logits}

    def target(self, batch):
        """
        Return a dictionary with:
        "label": the node-level target
        "mask": a boolean mask indicating which nodes are labeled
        "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        device = pred.device if hasattr(pred, 'device') else (
        pred["pred"].device if isinstance(pred, dict) and "pred" in pred else 
        (pred[0].device if isinstance(pred, list) and len(pred) > 0 else torch.device("cpu"))
        )

        # Move target to the same device as pred
        _target = _target.to(device)
        labeled = labeled.to(device)
        _size = _size.to(device)

        if isinstance(self.metric, dict):
            m = list(self.metric.keys())
        elif isinstance(self.metric, (list, tuple)):
            m = list(self.metric)
        else:
            m = self.metric

        if m == "macro_acc":
            pred = torch.cat(pred, dim=0)
            pred_argmax = pred[labeled].argmax(dim=-1)
            correct = (pred_argmax == _target[labeled]).float()
            score = functional.variadic_mean(correct, _size).mean()
            metric["macro_acc"] = score.item()

        else:
            raise ValueError(f"Unknown metric `{m}`")

        return metric
    

@R.register("tasks.EarlyExitClassification_walltime_ProtBert")
class EarlyExitClassification_walltime_ProtBert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(EarlyExitClassification_walltime_ProtBert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer
        self.num_class = num_class

    # ------------- helper to space‑separate sequences ------------
    @staticmethod
    def _prep_protbert(seqs):               # EDIT: name
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    # --------------------------- PREDICT -------------------------
    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()

        graphs = batch["graph"]

        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        batch_size = len(sequences)

        # ---- tokenise ----
        prepared = self._prep_protbert(sequences)
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,                # ProtBERT was trained up to 1024
            return_tensors="pt",
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # ---- temperatures / bookkeeping ----
        n_layers = self.model.config.num_hidden_layers
        temps    = torch.ones(n_layers, device=device)
        threshold = float(os.getenv("THRESHOLD", "0.0"))

        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob    = torch.full((batch_size,), -float("inf"), device=device)
        best_logits  = [None] * batch_size
        best_layers  = [None] * batch_size
        computed_layers = [None] * batch_size  
        active       = torch.arange(batch_size, device=device)

        # ---- build extended attention mask like BERT does ----
        ext_mask = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        # ---- initial embeddings ----
        hs = self.model.embeddings(input_ids)

        # ---- iterate through encoder layers (no weight sharing) ----
        for layer_idx, layer in enumerate(self.model.encoder.layer):     # EDIT: simple loop
            if len(active) == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = layer_idx

            # slice to active samples only
            hs_active = hs[active]

            # BERT layer forward
            out = layer(
                hs_active,
                attention_mask=ext_mask[active],
                head_mask=None,
                output_attentions=False
            )
            hs_active = out[0]                 # first element is hidden states
            hs[active] = hs_active

            # ---------- classifier ----------
            pooled   = hs_active.mean(dim=1)
            logits   = self.mlp[layer_idx](pooled)
            prob     = torch.sigmoid(logits / temps[layer_idx])
            max_p, _ = prob.max(dim=1)

            # ---------- best‑so‑far ----------
            is_final = layer_idx == n_layers - 1
            better   = max_p > best_prob[active]
            if is_final and os.getenv("SELECT_LAST") == "True":
                better = torch.ones_like(better, dtype=torch.bool)

            if better.any():
                g_idx = active[better]
                best_prob[g_idx] = max_p[better]
                for j, gi in enumerate(g_idx.tolist()):
                    best_logits[gi] = logits[better][j]
                    best_layers[gi] = layer_idx

            # ---------- early‑exit ----------
            exit_mask  = max_p > threshold
            newly_exit = active[exit_mask]
            still_act  = active[~exit_mask]

            for j, gi in enumerate(newly_exit.tolist()):
                final_logits[gi] = logits[exit_mask][j]
                final_layers[gi] = layer_idx

            active = still_act

        # ---- force exit any stragglers ----
        for gi in active.tolist():
            final_logits[gi] = best_logits[gi]
            final_layers[gi] = best_layers[gi]

        preds = torch.stack(final_logits, dim=0)

        # legacy 2000‑wide ascii tensor (unchanged)
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": preds,
            "layers": torch.tensor(final_layers, device=device, dtype=torch.int64),
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64),
            "sequences": ascii_mat,
        }

    # ------------- unchanged target / evaluate -----------------
    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        layers = preds["layers"]
        computed_layers = preds["computed_layers"]
        sequences = preds["sequences"]
        target = target.to(pred.device)

        metric_out = {}
        m = self.metric
        if m == "f1_max":
            score = metrics.f1_max(pred, target)
            metric_out[m] = score.item()
        else:
            raise ValueError(f"Unknown metric {m}")

        freq = torch.bincount(layers.cpu())
        avg_layer = (torch.arange(len(freq), device=freq.device) * freq).sum() / freq.sum()

        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed


        result["avg_layer"] = avg_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        result[m] = metric_out[m]
        return result

@R.register("tasks.EarlyExitProperty_continuous_protbert")
class EarlyExitProperty_continuous_protbert(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(EarlyExitProperty_continuous_protbert, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def predict(self, batch, all_loss=None, metric=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()        # ensure correct GPU
        self.mlp.to(device).eval()
        #print(f"device {device}")
        graphs  = batch["graph"]

        # ---- graphs → raw sequences -------------------------------------
        seqs = ["".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
                for g in graphs]
        B = len(seqs)

        # ---- temperatures / threshold ------------------------------------
        thr = float(os.getenv("THRESHOLD", "0.0"))
        tmp_file = os.getenv("TEMPERATURE_FILE")
        if tmp_file and tmp_file.lower() != "none":
            temps = torch.tensor(self._extract_temperatures(tmp_file), device=device)
        else:
            temps = torch.ones(self.num_layers, device=device)

        # ---- tokenizer ----------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache")
            )
        enc = self._tokenizer(
            self._prep_protbert(seqs),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt"
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]

        # ---- initial embeddings ------------------------------------------
        hs = self.model.embeddings(input_ids)                 # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0

        # ---- bookkeeping -------------------------------------------------
        final_log = [None] * B
        final_lay = [None] * B
        best_prob = torch.full((B,), -float("inf"), device=device)
        best_log  = [None] * B
        best_lay  = [None] * B
        computed_layers = [None] * B 
        active    = torch.arange(B, device=device)
         # ---- iterate through 30 encoder layers (no sharing) --------------
        for lidx, layer in enumerate(self.model.encoder.layer):
            if active.numel() == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = lidx 
            # forward only for still‑active samples
            hs_act = layer(
                hs[active],
                attention_mask=ext_mask[active],
                head_mask=None,
                output_attentions=False
            )[0]
            hs[active] = hs_act

            # task head
            pooled = hs_act.mean(dim=1)
            logits = self.mlp[lidx](pooled)
            prob   = torch.sigmoid(logits / temps[lidx])
            max_p  = prob.max(dim=1).values

            # best‑so‑far update
            is_final = lidx == self.num_layers - 1
            better   = max_p > best_prob[active]
            if is_final and os.getenv("SELECT_LAST", "False") == "True":
                better = torch.ones_like(better, dtype=torch.bool)
            if better.any():
                gi = active[better]
                best_prob[gi] = max_p[better]
                for k, gidx in enumerate(gi.tolist()):
                    best_log[gidx] = logits[better][k]
                    best_lay[gidx] = lidx

            # early‑exit decision
            exit_mask = max_p > thr
            newly_exit, still = active[exit_mask], active[~exit_mask]

            for k, gidx in enumerate(newly_exit.tolist()):
                final_log[gidx] = logits[exit_mask][k]
                final_lay[gidx] = lidx

            active = still

        # ---- force‑exit leftovers ----------------------------------------
        for gidx in active.tolist():
            final_log[gidx] = best_log[gidx]
            final_lay[gidx] = best_lay[gidx]

        preds = torch.stack(final_log, dim=0)                 # (B,C)

        # legacy 2000‑wide ASCII tensor
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in seqs],
            batch_first=True, padding_value=0
        )
        if ascii_mat.size(1) < 2000:
            ascii_mat = torch.cat(
                [ascii_mat, ascii_mat.new_zeros(B, 2000 - ascii_mat.size(1))],
                dim=1
            )

        return {
            "pred": preds,
            "layers": torch.tensor(final_lay, device=device, dtype=torch.int64),
            "sequences": ascii_mat, 
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64)
        }

 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        layers = preds["layers"]
        sequences = preds["sequences"]
        print(f"{pred.shape} pred.shape")
        print(f"self.num class {self.num_class}")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        target = target.to(device)
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score.item()
                metric["acc"] = acc
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)

        
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total
        metric["avg_layer"] = average_layer.item()

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()


        return metric









@R.register("tasks.EarlyExitClassificationTemperature_Node_continuous_ProtBert")
class EarlyExitClassificationTemperature_Node_continuous_ProtBert(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(EarlyExitClassificationTemperature_Node_continuous_ProtBert, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    @staticmethod
    def _prep_protbert(seqs):
        """Space‑separate, upper‑case, map U/O→X (ProtBERT convention)."""
        out = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            out.append(" ".join(list(s)))
        return out

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Same inputs/outputs as the original ESM‑2 version, but revised to use
        ProtBERT.  The overall control‑flow, early‑exit logic, and return
        structure are unchanged so downstream code keeps working.
        """
        # ---------------------------------------------------------------
        # 0) bookkeeping & set‑up
        # ---------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device).eval()
        self.mlp.to(device).eval()

        graphs = batch["graph"]
        sequences = [
            "".join(data.Protein.id2residue_symbol[r] for r in g.residue_type.tolist())
            for g in graphs
        ]
        B = len(sequences)

        # ---------------------------------------------------------------
        # 1) temperatures / thresholds
        # ---------------------------------------------------------------
        threshold = float(os.getenv("THRESHOLD"))
        percent   = float(os.getenv("PERCENT"))

        n_layers = getattr(self, "num_layers", self.model.config.num_hidden_layers)
        temps = torch.ones(n_layers, device=device)

        # ---------------------------------------------------------------
        # 2) tokenize with ProtBERT tokenizer (slow‑tokenizer, no lower‑case)
        # ---------------------------------------------------------------
        if not hasattr(self, "_tokenizer"):
            from transformers import AutoTokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(
                "Rostlab/prot_bert", do_lower_case=False, use_fast=False,
                cache_dir=os.getenv("HF_CACHE", "/scratch/anna19/hf_cache"),
            )

        enc = self._tokenizer(
            self._prep_protbert(sequences),
            add_special_tokens=True,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        input_ids, attn_mask = enc["input_ids"], enc["attention_mask"]  # (B,L)

        # ---------------------------------------------------------------
        # 3) initial embeddings (B,L,H) and Bert‑style attention mask
        # ---------------------------------------------------------------
        hs = self.model.embeddings(input_ids)                               # (B,L,H)
        ext_mask = (1.0 - attn_mask[:, None, None, :]) * -10000.0          # (B,1,1,L)

        # ---------------------------------------------------------------
        # 4) placeholders for results & trackers (match original output)
        # ---------------------------------------------------------------
        final_logits = [None] * B
        final_layers = torch.full((B,), -1, device=device)
        best_logits  = [None] * B
        best_prob    = torch.full((B,), -float("inf"), device=device)
        best_layers  = torch.full((B,), -1, device=device)
        computed_layers = torch.full((B,), -1, device=device) 
        active       = torch.arange(B, device=device)

        n_layers = getattr(self, "num_layers", self.model.config.num_hidden_layers)

        # ---------------------------------------------------------------
        # 5) iterate over ProtBERT encoder layers
        # ---------------------------------------------------------------
        for lidx, layer in enumerate(self.model.encoder.layer):
            if active.numel() == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = lidx 

            # ---- forward pass ONLY for active samples -----------------
            hs_act = layer(
                hs[active],                   # (A,L,H)
                attention_mask=ext_mask[active],
                head_mask=None,
                output_attentions=False,
            )[0]
            hs[active] = hs_act               # write‑back

            # ---- MLP head per sample ----------------------------------
            logits_list = []
            max_prob_vec = torch.empty(active.size(0), device=device)

            for loc, gidx in enumerate(active.tolist()):
                seq_len = len(sequences[gidx])
                # slice off the [CLS] token at pos 0; keep only residues
                h_i = hs_act[loc, 1 : seq_len + 1, :]      # (L_res,H)
                log_i = self.mlp[lidx](h_i)                # (L_res,C)
                logits_list.append(log_i)

                probs = torch.sigmoid(log_i / temps[lidx])
                max_prob_vec[loc] = probs.max(dim=1).values.mean()

                if (probs.max(dim=1).values > threshold).float().mean() >= percent:
                    final_logits[gidx] = log_i
                    final_layers[gidx] = lidx

            # ---- split finished vs. still active ----------------------
            done_mask   = final_layers[active] != -1
            newly_done  = active[done_mask]
            still_active = active[~done_mask]
            is_final    = lidx == n_layers - 1

            if still_active.numel() > 0:
                better = max_prob_vec[~done_mask] > best_prob[still_active]
                if is_final and os.getenv("SELECT_LAST") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    upd_idx = still_active[better]
                    best_prob[upd_idx]   = max_prob_vec[~done_mask][better]
                    best_layers[upd_idx] = lidx
                    # map local → global for logits list
                    for k, g in enumerate(upd_idx.tolist()):
                        best_logits[g] = logits_list[(~done_mask).nonzero(as_tuple=True)[0][k]]

            active = still_active

        # ---------------------------------------------------------------
        # 6) force‑exit any leftovers -----------------------------------
        for g in active.tolist():
            final_logits[g] = best_logits[g]
            final_layers[g] = best_layers[g]

        # ---------------------------------------------------------------
        # 7) legacy ASCII‑matrix (2000‑wide) ----------------------------
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True,
            padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": final_logits,               # list(Tensor[L_i,C])
            "layers": final_layers,             # Tensor[B]
            "sequences": ascii_mat,            # Tensor[B,2000]
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64)
        }

    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        layers = preds["layers"]
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            metric["layer"] = average_layer.item()

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()

        result = {}

        # with open(os.getenv("RESULT_PICKLE"), "wb") as f:
        #     pickle.dump(
        #         {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer,
        #          "metric": metric[_metric], "sequences": preds["sequences"]}, f
        #     )

        result["avg_layer"] = average_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        result["macro_acc"] = metric["macro_acc"]
        return result



##############################################################
### ESM ###
##############################################################

@R.register("tasks.Classification_walltime_ESM")
class Classification_walltime_ESM(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('f1_max'), verbose=0, num_class=1, weight=None, confidence_threshold=None):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(Classification_walltime_ESM, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.confidence_threshold = confidence_threshold
        self.metric = metric
        self.num_class = num_class

    def predict(self, batch, all_loss=None, metric=None):

        device = self.device
        graphs = batch["graph"]

        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)

        # 1) Tokenize once
        data_ = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)

        # 2) Build a padding mask
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)
        # If your sequences are right-padded, we can pass `padding_mask` to the transformer

        # 3) **Replicate ESM2's forward logic** EXACTLY

        # 3a) Embedding scale
        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)

        # 3b) Token dropout, if ESM2 is using it
        # (Check self.model.model.token_dropout)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12  # example
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            # avoid divide-by-zero for any empty sequences
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            scale_factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
            x = x * scale_factor.unsqueeze(-1).unsqueeze(-1)

        # 3c) If token != padding, multiply by 1 - padding_mask, etc.
        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # 3d) Now we do the same as ESM2: x => (T, B, E)
        x = x.transpose(0, 1)  # shape: [seq_len, batch_size, hidden_dim]

        # 6) Iterate over each layer exactly once
        layer_stop = int(os.getenv("LAYER"))
        for layer_idx, layer_module in enumerate(self.model.model.layers):

            # apply the layer
            layer_out = layer_module(
                x,
                self_attn_padding_mask=padding_mask
                    if padding_mask is not None else None,
                need_head_weights=False
            )
            # Some ESM versions return (hidden, attn), some just hidden
            if isinstance(layer_out, tuple):
                layer_out = layer_out[0]

            # Place updated states back
            x = layer_out

            # Check if this is the **final** layer
            is_final_layer = (layer_idx == self.model.model.num_layers - 1)
            if is_final_layer:
                # ESM2 does a final layer norm after the loop
                x = self.model.model.emb_layer_norm_after(x)

            if layer_idx == layer_stop:
                hs_for_mlp = x.transpose(0, 1)  # => [num_active, seq_len, hidden_dim]
                mlp_input = hs_for_mlp.mean(dim=1)  # shape [num_active, hidden_dim]
                logits = self.mlp[layer_idx](mlp_input)
                break
        return{"pred": logits}
        
    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)
        labeled = ~torch.isnan(target)
        metric_out = {}
        if isinstance(self.metric, dict):
            m = list(self.metric.keys())
        else:
            m = self.metric

        if m == "auroc@micro":
            score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
        elif m == "auprc@micro":
            score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
        elif m == "f1_max":
            score = metrics.f1_max(pred, target)
        elif m == "acc" or "acc" in m:
            m = "acc"
            score = []
            num_class = 0
            for i, cur_num_class in enumerate(self.num_class):
                _pred = pred[:, num_class:num_class + cur_num_class]
                _target = target[:, i]
                _labeled = labeled[:, i]
                _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                score.append(_score)
                num_class += cur_num_class
            score = torch.stack(score)
        else:
            raise ValueError(f"Unknown metric {m}")
        metric_out[m] = score.item()
        return metric_out


@R.register("tasks.EarlyExitClassification_walltime")
class EarlyExitClassification_walltime(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, confidence_threshold=None):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(EarlyExitClassification_walltime, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.confidence_threshold = confidence_threshold
        self.metric = metric
        self.num_class = num_class

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures


    def predict(self, batch, all_loss=None, metric=None):

        device = self.device
        graphs = batch["graph"]

        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)

        batch_size = len(sequences)

        # We'll store final chosen logits + chosen layer
        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob     = torch.full((batch_size,), -float("inf"), device=device)   # NEW
        best_logits   = [None] * batch_size                                       # NEW
        best_layers   = [None] * batch_size
        computed_layers = [None] * batch_size                                       

        # 1) Tokenize once
        data_ = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)

        # 2) Build a padding mask
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)
        # If your sequences are right-padded, we can pass `padding_mask` to the transformer

        # 3) **Replicate ESM2's forward logic** EXACTLY

        # 3a) Embedding scale
        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)

        # 3b) Token dropout, if ESM2 is using it
        # (Check self.model.model.token_dropout)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)

            # ESM2 also does a ratio-based rescaling
            # See the official code block that looks like:
            #    mask_ratio_train = 0.15 * 0.8
            #    src_lengths = (~padding_mask).sum(-1)
            #    mask_ratio_observed = (tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            #    x = x * (1 - mask_ratio_train) / (1 - mask_ratio_observed)[:, None, None]
            mask_ratio_train = 0.12  # example
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            # avoid divide-by-zero for any empty sequences
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            scale_factor = (1 - mask_ratio_train) / (1 - mask_ratio_observed)
            x = x * scale_factor.unsqueeze(-1).unsqueeze(-1)

        # 3c) If token != padding, multiply by 1 - padding_mask, etc.
        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # 3d) Now we do the same as ESM2: x => (T, B, E)
        x = x.transpose(0, 1)  # shape: [seq_len, batch_size, hidden_dim]

        # 4) Keep track of "active" sample indices
        active_indices = torch.arange(batch_size, device=device)

        # 5) Temperatures
        threshold = float(os.getenv("THRESHOLD"))
        temperature_file = os.getenv("TEMPERATURE_FILE")
        if temperature_file is not None and temperature_file != 'None':
            temperatures = self.extract_temperatures(temperature_file)
            temperatures = torch.tensor(temperatures, device=device)
        else:
           temperatures = torch.ones(33)


        # 6) Iterate over each layer exactly once
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            if len(active_indices) == 0:
                break
            for idx in active_indices.tolist():
                computed_layers[idx] = layer_idx

            # Gather x for active only
            # x shape is [seq_len, batch_size, hidden_dim].
            # We want to slice out "batch_size" dimension = active_indices
            # We'll do an index_select along dim=1:
            hs_active = x[:, active_indices, :]

            # apply the layer
            layer_out = layer_module(
                hs_active,
                self_attn_padding_mask=padding_mask[active_indices]
                    if padding_mask is not None else None,
                need_head_weights=False
            )
            # Some ESM versions return (hidden, attn), some just hidden
            if isinstance(layer_out, tuple):
                hs_active = layer_out[0]
            else:
                hs_active = layer_out

            # Place updated states back
            x[:, active_indices, :] = hs_active

            # Check if this is the **final** layer
            is_final_layer = (layer_idx == self.model.model.num_layers - 1)
            if is_final_layer:
                # ESM2 does a final layer norm after the loop
                hs_active = self.model.model.emb_layer_norm_after(hs_active)

            # We want to feed the representation to an MLP
            # Usually ESM2 store "representations[layer_idx+1]" as hs_active.transpose(0,1)
            # But let's just do the same for MLP
            hs_for_mlp = hs_active.transpose(0, 1)  # => [num_active, seq_len, hidden_dim]

            # If normal classification used mean-pooling for each sample, do it here:
            mlp_input = hs_for_mlp.mean(dim=1)  # shape [num_active, hidden_dim]

            # Then apply the layer's MLP
            logits_active = self.mlp[layer_idx](mlp_input)

            # Apply temperature scaling & threshold
            scaled_logits = logits_active / temperatures[layer_idx]
            probabilities = torch.sigmoid(scaled_logits)
            max_prob, _ = probabilities.view(probabilities.size(0), -1).max(dim=1)

            # ---------- NEW: keep the best prob/logits seen so far ----------
            better = max_prob > best_prob[active_indices]
            if is_final_layer and os.environ.get("SELECT_LAST", "False") == "True":
                better = torch.full_like(best_prob[active_indices], fill_value=True, dtype=torch.bool)

            if better.any():
                idx_global = active_indices[better]          # indices in the original batch
                best_prob[idx_global]   = max_prob[better]   # update tensor (in‑place assignment)
                for g in idx_global.tolist():                # ✅ update Python lists
                    best_layers[g] = layer_idx
                # store **raw** logits (same as you already return when a sample exits)
                best_logits_arr = logits_active[better]      # shape [n_better, num_classes]
                for j, g in enumerate(idx_global.tolist()):
                    best_logits[g] = best_logits_arr[j]
            # ----------------------------------------



            meet_threshold_mask = (max_prob > threshold)

            newly_exited = active_indices[meet_threshold_mask]
            still_active = active_indices[~meet_threshold_mask]

            # Save final logits/layer for those who exit
            for i, global_idx in enumerate(newly_exited.tolist()):
                final_logits[global_idx] = logits_active[meet_threshold_mask][i]
                final_layers[global_idx] = layer_idx

            # Update active_indices
            active_indices = still_active

        # 7) If any remain after the final layer, they're forced to exit
        if len(active_indices) > 0:
            # we already computed final layer above (with LN),
            # so let's apply the final MLP again for them, if needed
            for g in active_indices.tolist():
                final_logits[g] = best_logits[g]
                final_layers[g] = best_layers[g]


        # 8) Stack results
        selected_outputs = torch.stack(final_logits, dim=0)

        encoded_sequences = []
        max_len = 2000
        for seq in sequences:
            ascii_ids = [ord(c) for c in seq]
            padded = ascii_ids + [0] * (max_len - len(seq))
            encoded_sequences.append(padded)

        return {"pred":selected_outputs, "layers":torch.tensor(final_layers, device=self.device, dtype=torch.int64), "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64), "sequences":torch.tensor(encoded_sequences, device=self.device, dtype=torch.int64)} 

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        layers = preds["layers"]
        computed_layers = preds["computed_layers"]
        sequences = preds["sequences"]
        target = target.to(pred.device)

        metric_out = {}
        m = self.metric
        if m == "f1_max":
            score = metrics.f1_max(pred, target)
            metric_out[m] = score.item()
        else:
            raise ValueError(f"Unknown metric {m}")

        freq = torch.bincount(layers.cpu())
        avg_layer = (torch.arange(len(freq), device=freq.device) * freq).sum() / freq.sum()

        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed


        result["avg_layer"] = avg_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        result[m] = metric_out[m]
        return result


@R.register("tasks.EarlyExitProperty_continuous")
class EarlyExitProperty_continuous(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(EarlyExitProperty_continuous, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures
    
    def predict(self, batch, all_loss=None, metric=None):
        device = self.device
        threshold = float(os.getenv("THRESHOLD"))
        #threshold = 0
        #temperature_file = "/shared/nas2/anna19/early_exit/esm-s/temperature_files/protein01_test_02.csv"
        temperature_file = os.getenv("TEMPERATURE_FILE")
        if temperature_file is not None and temperature_file != 'None':
            temperatures = self.extract_temperatures(temperature_file)
            temperatures = torch.tensor(temperatures, device=device)
        else:
            temperatures = torch.ones(self.num_layers, device=device)

        
        graphs = batch["graph"]
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        batch_size = len(sequences)
        input = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(input)
        batch_tokens = batch_tokens.to(device)
        
        # ---------- ESM-2 embedding (same as Classification version) ----------
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        # x = x + self.model.model.embed_positions(batch_tokens)
        # x = self.model.model.emb_layer_norm_before(x)

        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob     = torch.full((batch_size,), -float("inf"), device=device)   # NEW
        best_logits   = [None] * batch_size                                       # NEW
        best_layers   = [None] * batch_size
        computed_layers = [None] * batch_size  


        # token-dropout (if present)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            x = x * ((1 - mask_ratio_train) / (1 - mask_ratio_observed)).unsqueeze(-1).unsqueeze(-1)

        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # ESM expects [T,B,E]
        x = x.transpose(0, 1)
        active_indices = torch.arange(batch_size, device=device)
       #print(f"{x.shape}")
        #print(f"padding mask shape {padding_mask.shape}")
        # ---------------------------------------------------------------------
        logits_per_layer = []
        for layer_idx, layer_module in enumerate(self.model.model.layers):
            if len(active_indices) == 0:
                break
            for idx in active_indices.tolist():
                computed_layers[idx] = layer_idx 
            hs_active = x[:, active_indices, :]
            layer_out = layer_module(
                hs_active,
                self_attn_padding_mask=padding_mask[active_indices]
                    if padding_mask is not None else None,
                need_head_weights=False,
            )

            hs_active = layer_out[0]
            #print(f"hs_active.shape {len(hs_active)} {hs_active[0].shape} {hs_active[1].shape}")

            x[:, active_indices, :] = hs_active

            is_final = (layer_idx == self.model.model.num_layers - 1)
            if is_final:
                hs_active = self.model.model.emb_layer_norm_after(hs_active)

            mlp_input = hs_active.transpose(0,1)
            mlp_input = mlp_input.mean(dim=1) ##adding mean pooling MODIFICATIOn
            logits = self.mlp[layer_idx](mlp_input)
            
            scaled_logits = logits / temperatures[layer_idx]
            probabilities = torch.sigmoid(scaled_logits)
            max_prob, _ = probabilities.view(probabilities.size(0), -1).max(dim=1)

            better = max_prob > best_prob[active_indices]
            if is_final and os.environ.get("SELECT_LAST", "False") == "True":
                better = torch.full_like(best_prob[active_indices], fill_value=True, dtype=torch.bool)
            if better.any():
                idx_global = active_indices[better]
                best_prob[idx_global] = max_prob[better]
                best_logits_arr = logits[better]
                for j, g in enumerate(idx_global.tolist()):
                    best_logits[g] = best_logits_arr[j]
                    best_layers[g] = layer_idx
            
            meet_threshold_mask = (max_prob > threshold)
            newly_exited = active_indices[meet_threshold_mask]
            active_indices = active_indices[~meet_threshold_mask]  
            for i, global_idx in enumerate(newly_exited.tolist()):
                final_logits[global_idx] = logits[meet_threshold_mask][i]
                final_layers[global_idx] = layer_idx

             
        if len(active_indices) > 0:
            for g in active_indices.tolist():
                final_logits[g] = best_logits[g]
                final_layers[g] = best_layers[g]

        selected_outputs = torch.stack(final_logits, dim=0)


        encoded_sequences = []
        max_len = 2000
        for seq in sequences:
            ascii_ids = [ord(c) for c in seq]
            padded = ascii_ids + [0] * (max_len - len(seq))
            encoded_sequences.append(padded)

        
        #print(f"selected_outputs {selected_outputs.shape}")
        #selected_outputs = torch.stack(final_logits, dim=0)

        return {"pred":selected_outputs, "layers":torch.tensor(final_layers, device=self.device, dtype=torch.int64), "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64), "sequences":torch.tensor(encoded_sequences, device=self.device, dtype=torch.int64)} 

 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        pred = preds["pred"]
        layers = preds["layers"]
        sequences = preds["sequences"]
        computed_layers = preds["computed_layers"]
        print(f"{pred.shape} pred.shape")
        print(f"self.num class {self.num_class}")
        labeled = ~torch.isnan(target)
        metric = {}
        for _metric in self.metric:
            if _metric == "auroc@micro":
                score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
            elif _metric == "auprc@micro":
                score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
            elif _metric == "f1_max":
                score = metrics.f1_max(pred, target)
            elif _metric == "acc":
                score = []
                num_class = 0
                for i, cur_num_class in enumerate(self.num_class):
                    _pred = pred[:, num_class:num_class + cur_num_class]
                    _target = target[:, i]
                    _labeled = labeled[:, i]
                    _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                    score.append(_score)
                    num_class += cur_num_class
                score = torch.stack(score)
                acc = score.item()
                metric["acc"] = acc
            else:
                raise ValueError("Unknown criterion `%s`" % _metric)
            

        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total
        metric["avg_layer"] = average_layer.item()
        

        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()

        return metric


@R.register("tasks.NormalProperty_continuous")
class NormalProperty_continuous(tasks.Task, core.Configurable):
    eps = 1e-10
    _option_members = {"task", "criterion", "metric"}

    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super(NormalProperty_continuous, self).__init__()
        self.model = model
        self.task = task
        self.criterion = criterion
        self.metric = metric
        self.num_mlp_layer = num_mlp_layer
        # For classification tasks, we disable normalization tricks.
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_class = (num_class,) if isinstance(num_class, int) else num_class
        self.num_layers = model.num_layers
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_dropout = mlp_dropout
        self.graph_construction_model = graph_construction_model
        self.verbose = verbose
        self.confidence_threshold = confidence_threshold
        for param in self.model.parameters():
            param.requires_grad = False


    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures
    
    def predict(self, batch, all_loss=None, metric=None):

        device = self.device
        graphs = batch["graph"]
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)
        batch_size = len(sequences)
        input = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(input)
        batch_tokens = batch_tokens.to(device)
        
        # ---------- ESM-2 embedding (same as Classification version) ----------
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        # x = x + self.model.model.embed_positions(batch_tokens)
        # x = self.model.model.emb_layer_norm_before(x)
        layer_stop = int(os.getenv("LAYER"))
        # token-dropout (if present)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_lengths = (~padding_mask).sum(-1)
            mask_ratio_observed = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_lengths
            mask_ratio_observed = torch.clamp(mask_ratio_observed, min=1e-9)
            x = x * ((1 - mask_ratio_train) / (1 - mask_ratio_observed)).unsqueeze(-1).unsqueeze(-1)

        if padding_mask.any():
            x = x * (1 - padding_mask.unsqueeze(-1).type_as(x))

        # ESM expects [T,B,E]
        x = x.transpose(0, 1)
        # ---------------------------------------------------------------------

        for layer_idx, layer_module in enumerate(self.model.model.layers):
            layer_out = layer_module(
                x,
                self_attn_padding_mask=padding_mask
                    if padding_mask is not None else None,
                need_head_weights=False,
            )
            if isinstance(layer_out, tuple):
                x = layer_out[0]
            else:
                x = layer_out

            is_final = layer_idx == self.model.model.num_layers - 1
            if is_final:
                x = self.model.model.emb_layer_norm_after(x)

            if layer_idx == layer_stop:
                mlp_input = x.transpose(0,1)
                mlp_input = mlp_input.mean(dim=1) ##adding mean pooling MODIFICATIOn
                logits = self.mlp[layer_idx](mlp_input)

        return {"pred":logits} 
 
    def target(self, batch):
        target = torch.stack([batch[t].float() for t in self.task], dim=-1)
        labeled = batch.get("labeled", torch.ones(len(target), dtype=torch.bool, device=target.device))
        target[~labeled] = math.nan
        return target

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)
        labeled = ~torch.isnan(target)
        metric_out = {}
        if isinstance(self.metric, dict):
            m = list(self.metric.keys())
        else:
            m = self.metric

        if m == "auroc@micro":
            score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
        elif m == "auprc@micro":
            score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
        elif m == "f1_max":
            score = metrics.f1_max(pred, target)
        elif m == "acc" or "acc" in m:
            m = "acc"
            score = []
            num_class = 0
            for i, cur_num_class in enumerate(self.num_class):
                _pred = pred[:, num_class:num_class + cur_num_class]
                _target = target[:, i]
                _labeled = labeled[:, i]
                _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                score.append(_score)
                num_class += cur_num_class
            score = torch.stack(score)
        else:
            raise ValueError(f"Unknown metric {m}")
        metric_out[m] = score.item()
        return metric_out


@R.register("tasks.EarlyExitClassificationTemperature_Node_continuous")
class EarlyExitClassificationTemperature_Node_continuous(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(EarlyExitClassificationTemperature_Node_continuous, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def extract_temperatures(self, file_path):
        temperatures = []
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip the header row
            for row in reader:
                # Extract the tensor value from the second column and parse the float
                tensor_string = row[1]
                value = float(tensor_string.split('(')[1].split(',')[0])
                temperatures.append(value)
        return temperatures

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Return a list of predictions (one per layer).
        Each layer's node_feature is passed to a separate MLP.
        """

        graphs = batch["graph"]
        sequences = []
        device = next(self.model.parameters()).device
        n_layers  = 33

        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)

        B = len(sequences)
        temp_file = os.getenv("TEMPERATURE_FILE")
        if temp_file and temp_file.lower() != "none":
            temps = torch.tensor(self.extract_temperatures(temp_file), device=device)              # shape (num_layers,)
        else:
            temps = torch.ones(self.model.model.num_layers, device=device)

        threshold = float(os.getenv("THRESHOLD"))
        percent = float(os.getenv("PERCENT"))

        final_logits = [None] * B
        final_layers = torch.full((B,), -1, device=device) 
        best_logits = [None] * B
        best_prob = torch.full((B,), -float("inf"), device=device)
        best_layers = torch.full((B,), -1, device=device) 
        computed_layers = torch.full((B,), -1, device=device)

        data_ = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_len = (~padding_mask).sum(-1)
            mask_ratio_obs = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_len
            mask_ratio_obs = torch.clamp(mask_ratio_obs, min=1e-9)
            x *= ((1 - mask_ratio_train) / (1 - mask_ratio_obs)).unsqueeze(-1).unsqueeze(-1)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x)) 
        x = x.transpose(0, 1) 
        active = torch.arange(B, device=device)

        total = 0

        for lidx, layer in enumerate(self.model.model.layers):
            if active.numel() == 0:
                break
            for idx in active.tolist():
                computed_layers[idx] = lidx 
            h = x[:, active, :]
            h = layer(h, self_attn_padding_mask=padding_mask[active], need_head_weights=False)[0]
            x[:, active, :] = h
            h_seq = h.transpose(0,1)
            logits_list = []
            max_prob_for_mean = torch.empty(active.size(0), device=device)
            for local_idx, glob_idx in enumerate(active.tolist()):
                seq_len = len(sequences[glob_idx])
                h_i = h_seq[local_idx, :seq_len, :]
                logits_i = self.mlp[lidx](h_i)
                logits_list.append(logits_i)
                scaled = logits_i/temps[lidx]
                probs = torch.sigmoid(scaled)
                max_res = probs.max(dim=1).values
                max_prob_for_mean[local_idx] = max_res.mean()

                if (max_res > threshold).float().mean() >= percent:
                    final_logits[glob_idx] = logits_i
                    final_layers[glob_idx] = lidx

            done_mask = final_layers[active] != -1
            newly_done = active[done_mask]
            still_active = active[~done_mask]
            is_final = lidx == n_layers - 1

            if still_active.numel() > 0:
                better = max_prob_for_mean[~done_mask] > best_prob[still_active]
                if is_final and os.getenv("SELECT_LAST", "False") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    upd_idx = still_active[better]
                    best_prob[upd_idx]   = max_prob_for_mean[~done_mask][better]
                    best_layers[upd_idx] = lidx
                    for k, g in enumerate(upd_idx.tolist()):
                        best_logits[g] = logits_list[(~done_mask).nonzero(as_tuple=True)[0][k]]
            active = still_active
        if active.numel() > 0:
            for g in active.tolist():
                final_logits[g] = best_logits[g]
                final_layers[g] = best_layers[g]

        ascii_mat = torch.nn.utils.rnn.pad_sequence(
                [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
                batch_first=True, padding_value=0,
            )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)
        return {
            "pred": final_logits,
            "layers": final_layers,
            "computed_layers": torch.tensor(computed_layers, device=self.device, dtype=torch.int64),
            "sequences": ascii_mat
        }
    
    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        layers = preds["layers"]
        layer_frequencies = torch.bincount(layers)
        total = layer_frequencies.sum()
        layer_indices = torch.arange(len(layer_frequencies), device=layer_frequencies.device)
        average_layer = (layer_indices * layer_frequencies).sum() / total

        for _metric in self.metric:
            if _metric in ["mae", "rmse"]:
                # Typically for regression
                if _metric == "mae":
                    score = F.l1_loss(pred, _target, reduction="none")
                else:  # rmse
                    score = F.mse_loss(pred, _target, reduction="none").sqrt()

                score = functional.masked_mean(score, labeled, dim=0)

            elif _metric in ["micro_auroc", "micro_auprc"]:
                # Single "micro" approach across all labeled nodes
                if _metric == "micro_auroc":
                    score = metrics.area_under_roc(pred[labeled], _target[labeled])
                else:
                    score = metrics.area_under_prc(pred[labeled], _target[labeled])

            elif _metric in ["macro_auroc", "macro_auprc"]:
                # "macro" means compute per-graph, then average
                if _metric == "macro_auroc":
                    score = metrics.variadic_area_under_roc(pred[labeled], _target[labeled], _size).mean()
                else:
                    score = metrics.variadic_area_under_prc(pred[labeled], _target[labeled], _size).mean()

            elif _metric == "macro_acc":
                # One typical approach for multi-class:
                # (pred[labeled].argmax(-1) == _target[labeled]).float()
                #print(f"labeled.shape {labeled.shape}")
                #print(f"target shape {_target.shape}")
                #pred = torch.cat(pred, dim=0)
                pred = torch.cat(pred, dim=0)
                #print(f"pred.shape {pred.shape}")
                pred_argmax = pred[labeled].argmax(dim=-1)
                correct = (pred_argmax == _target[labeled]).float()
                score = functional.variadic_mean(correct, _size).mean()
                metric["macro_acc"] = score.item()

            else:
                raise ValueError(f"Unknown metric `{_metric}`")
            metric["layer"] = average_layer.item()

        computed_layers = preds["computed_layers"]
        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed
        metric["avg_computed_layer"] = average_computed_layer.item()

        result = {}

        # with open(os.getenv("RESULT_PICKLE"), "wb") as f:
        #     pickle.dump(
        #         {"preds": pred, "target": target, "layers": layers, "avg_computed_layer": average_computed_layer,
        #          "metric": metric[_metric], "sequences": preds["sequences"]}, f
        #     )

        result["avg_layer"] = average_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        result["macro_acc"] = metric["macro_acc"]
        return result
    

@R.register("tasks.ClassificationTemperature_Node_continuous")
class ClassificationTemperature_Node_continuous(tasks.Task, core.Configurable):
    _option_members = {"criterion", "metric"}

    def __init__(
        self,
        model,
        criterion="bce",
        metric=("macro_auprc", "macro_auroc"),
        num_mlp_layer=1,
        normalization=True,
        num_class=None,
        verbose=0,
    ):
        super(ClassificationTemperature_Node_continuous, self).__init__()
        self.model = model
        self.criterion = criterion
        self.metric = metric
        # For classification tasks, disable normalization
        self.normalization = normalization and ("ce" not in criterion) and ("bce" not in criterion)
        self.num_mlp_layer = num_mlp_layer
        self.num_class = num_class
        self.verbose = verbose
        self.num_layers = 33

    def preprocess(self, train_set, valid_set, test_set):
        """
        Compute mean, std, and num_class on the training set,
        then build an MLP head per layer.
        """
        # Determine whether we are working at the node, atom, or residue level
        self.view = getattr(train_set[0]["graph"], "view", "atom")

        # Collect all target values from the train set for statistics
        values_list = []
        for data in train_set:
            values_list.append(data["graph"].target)  # shape: (num_nodes,) or (num_residues,)

        values = torch.cat(values_list, dim=0)
        mean = values.float().mean()
        std = values.float().std()

        # Figure out number of classes if doing classification
        num_class = 1
        if values.dtype == torch.long:
            # If max label is >1 or not using BCE, it means multiclass
            nmax = values.max().item()
            if nmax > 1 or "bce" not in self.criterion:
                nmax += 1
            num_class = nmax

        self.register_buffer("mean", torch.as_tensor(mean, dtype=torch.float))
        self.register_buffer("std", torch.as_tensor(std, dtype=torch.float))
        self.num_class = self.num_class or num_class

    def predict(self, batch, all_loss=None, metric=None):
        """
        Return a list of predictions (one per layer).
        Each layer's node_feature is passed to a separate MLP.
        """

        graphs = batch["graph"]
        sequences = []
        device = next(self.model.parameters()).device
        n_layers  = 33

        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            sequence = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(sequence)

        B = len(sequences)

        layer = int(os.getenv("LAYER"))

        data_ = [(f"protein_{i}", s) for i, s in enumerate(sequences)]
        _, _, batch_tokens = self.model.alphabet.get_batch_converter()(data_)
        batch_tokens = batch_tokens.to(device)
        padding_mask = batch_tokens.eq(self.model.model.padding_idx)

        x = self.model.model.embed_scale * self.model.model.embed_tokens(batch_tokens)
        if getattr(self.model.model, "token_dropout", False):
            mask_idx = self.model.model.mask_idx
            x.masked_fill_((batch_tokens == mask_idx).unsqueeze(-1), 0.0)
            mask_ratio_train = 0.12
            src_len = (~padding_mask).sum(-1)
            mask_ratio_obs = (batch_tokens == mask_idx).sum(-1).to(x.dtype) / src_len
            mask_ratio_obs = torch.clamp(mask_ratio_obs, min=1e-9)
            x *= ((1 - mask_ratio_train) / (1 - mask_ratio_obs)).unsqueeze(-1).unsqueeze(-1)
        x = x * (1 - padding_mask.unsqueeze(-1).type_as(x)) 
        x = x.transpose(0, 1) 
        
        for lidx, layer_mod in enumerate(self.model.model.layers):
            h = layer_mod(x, self_attn_padding_mask = padding_mask, need_head_weights=False)[0]
            x = h
            if lidx == layer:
                break

        h_seq = h.transpose(0,1)
        logits_list = []
        for b in range(B):
            seq_len   = len(sequences[b])
            h_b       = h_seq[b, :seq_len, :]              # strip pad positions
            logits_b  = self.mlp[layer](h_b)              # (seq_len, num_classes)
            logits_list.append(logits_b)

        return {"pred": logits_list}
            
    def target(self, batch):
        """
        Return a dictionary with:
          "label": the node-level target
          "mask": a boolean mask indicating which nodes are labeled
          "size": used for some metrics requiring per-graph aggregates
        """
        graph = batch["graph"]
        size = graph.num_nodes if self.view in ["node", "atom"] else graph.num_residues
        return {
            "label": graph.target,   # shape: (num_nodes,) or (num_residues,)
            "mask": graph.mask,      # shape: (num_nodes,) or (num_residues,)
            "size": size
        }

    def evaluate(self, preds, target):
        """
        Evaluate each layer's predictions given the `target`.
        preds: list of tensors, each is shape (N, num_class) or (N,) depending on your MLP output
        target: dict with { "label", "mask", "size" }
        """
        metric = {}
        _target = target["label"]
        _mask = target["mask"]
        labeled = ~torch.isnan(_target) & _mask
        _size = functional.variadic_sum(labeled.long(), target["size"])
        pred = preds["pred"]

        device = pred.device if hasattr(pred, 'device') else (
        pred["pred"].device if isinstance(pred, dict) and "pred" in pred else 
        (pred[0].device if isinstance(pred, list) and len(pred) > 0 else torch.device("cpu"))
        )

        # Move target to the same device as pred
        _target = _target.to(device)
        labeled = labeled.to(device)
        _size = _size.to(device)

        if isinstance(self.metric, dict):
            m = list(self.metric.keys())
        elif isinstance(self.metric, (list, tuple)):
            m = list(self.metric)
        else:
            m = self.metric
        if isinstance(m, list) and len(m) == 1:
            m = m[0]

        if m == "macro_acc" or "macro_acc" in m:
            pred = torch.cat(pred, dim=0)
            pred_argmax = pred[labeled].argmax(dim=-1)
            correct = (pred_argmax == _target[labeled]).float()
            score = functional.variadic_mean(correct, _size).mean()
            metric["macro_acc"] = score.item()

        else:
            raise ValueError(f"Unknown metric `{m}`")

        return metric


def evaluate_classification(preds, target, metric, num_class=None):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)
        labeled = ~torch.isnan(target)
        metric_out = {}
        if isinstance(metric, dict):
            m = list(metric.keys())
        else:
            m = metric

        if m == "auroc@micro":
            score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
        elif m == "auprc@micro":
            score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
        elif m == "f1_max":
            score = metrics.f1_max(pred, target)
        elif m == "acc" or "acc" in m:
            m = "acc"
            score = []
            num_class = 0
            for i, cur_num_class in enumerate(num_class):
                _pred = pred[:, num_class:num_class + cur_num_class]
                _target = target[:, i]
                _labeled = labeled[:, i]
                _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                score.append(_score)
                num_class += cur_num_class
            score = torch.stack(score)
        else:
            raise ValueError(f"Unknown metric {m}")
        metric_out[m] = score.item()
        layer_idx = int(os.getenv("LAYER"))
        out_file = os.getenv("OUT_FILE")
        
        # Load existing data if file exists, or create new structure
        if os.path.exists(out_file) and layer_idx >= 0:
            data = torch.load(out_file)
        else:
            data = {"preds_by_layer": [], "target": None}
        
        # Append this layer's predictions
        data["preds_by_layer"].append(pred.detach().cpu())
        
        # Store target once (first layer)
        if data["target"] is None:
            data["target"] = target.detach().cpu()
        
        # Save back to file
        torch.save(data, out_file)
        return metric_out

def evaluate_property_confidence(preds, target, self_num_class):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)

        out_file = os.getenv("OUT_FILE")
        layer_idx = int(os.getenv("LAYER"))

        if os.path.exists(out_file):
            data = torch.load(out_file, map_location="cpu")
            # If file exists but doesn't match expected structure, reset
            if not isinstance(data, dict) or "preds_by_layer" not in data:
                data = {"preds_by_layer": [], "target": None}
        else:
            data = {"preds_by_layer": [], "target": None}

        # Append this layer's logits
        data["preds_by_layer"].append(pred.detach().cpu())

        # Store target once (first write)
        if data.get("target") is None:
            data["target"] = target.detach().cpu()

        torch.save(data, out_file)

        labeled = ~torch.isnan(target)
        metric_out = {}
        m = "acc"
        score = []
        num_class = 0
        for i, cur_num_class in enumerate(self_num_class):
            _pred = pred[:, num_class:num_class + cur_num_class]
            _target = target[:, i]
            _labeled = labeled[:, i]
            _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
            score.append(_score)
            num_class += cur_num_class
        score = torch.stack(score)

        metric_out[m] = score.item()

        return metric_out

##### ProtAlbert ######

@R.register("tasks.Classification_walltime_ProtAlbert") #--> old, was used for first max
class Classification_walltime_ProtAlbert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(Classification_walltime_ProtAlbert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    @staticmethod
    def _prep_protalbert(seqs):
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #print(f"device {device}")
        graphs = batch["graph"]
        self.model.to(device).eval() 
        self.mlp.to(device).eval()


        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)
                              

        # 1) Tokenize once
        prepared = self._prep_protalbert(sequences)                       # EDIT
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,          # ProtAlbert max = 512
            return_tensors="pt",
            max_length = 550,
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)


        # 5) Temperatures
        layer_out = int(os.getenv("LAYER"))

        n_layers  = self.model.config.num_hidden_layers

        hs = self.model.embeddings(input_ids)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)
        attn_ext = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        groups = self.model.encoder.albert_layer_groups
        layers_per_group = n_layers // self.model.config.num_hidden_groups
        global_layer_idx = 0
        layer_int = 0
        for group in groups:
            for _ in range(layers_per_group):

                # ---- forward one logical layer ----
                                               # (N,L,H)
                hs = group(
                    hs,
                    attention_mask=attn_ext,
                    head_mask=[None] * self.model.config.num_hidden_layers,  # ← FIX HERE
                    output_attentions=False,
                )
                hs = hs[0] if isinstance(hs, tuple) else hs

                # ---- classifier ----
                if layer_int == layer_out:
                    pooled = hs.mean(dim=1)                                        # (N,H)
                    logits = self.mlp[global_layer_idx](pooled)
                layer_int += 1
                global_layer_idx += 1
        return {
            "pred": logits,
        }

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        target = target.to(pred.device)
        labeled = ~torch.isnan(target)
        metric_out = {}
        if isinstance(self.metric, dict):
            m = list(self.metric.keys())
        else:
            m = self.metric

        if m == "auroc@micro":
            score = metrics.area_under_roc(pred.flatten(), target.long().flatten())
        elif m == "auprc@micro":
            score = metrics.area_under_prc(pred.flatten(), target.long().flatten())
        elif m == "f1_max":
            score = metrics.f1_max(pred, target)
        elif m == "acc" or "acc" in m:
            m = "acc"
            score = []
            num_class = 0
            for i, cur_num_class in enumerate(self.num_class):
                _pred = pred[:, num_class:num_class + cur_num_class]
                _target = target[:, i]
                _labeled = labeled[:, i]
                _score = metrics.accuracy(_pred[_labeled], _target[_labeled].long())
                score.append(_score)
                num_class += cur_num_class
            score = torch.stack(score)
        else:
            raise ValueError(f"Unknown metric {m}")
        metric_out[m] = score.item()
        return metric_out

@R.register("tasks.EarlyExitClassification_walltime_ProtAlbert") #--> old, was used for first max
class EarlyExitClassification_walltime_ProtAlbert(tasks.Task, core.Configurable):
    def __init__(self, model, metric=('auprc@micro', 'f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        """
        Args:
            model_checkpoint (str): Path to the saved model checkpoint.
            mlp_layers (nn.ModuleList): MLP modules for each layer.
            confidence_classifier (nn.Module): Confidence classifier.
            confidence_threshold (float): Threshold for early exit based on confidence.
        """
        super(EarlyExitClassification_walltime_ProtAlbert, self).__init__()
        self.model = model  # Load the main model from checkpoint
        self.metric = metric
        self.tokenizer=tokenizer

    @staticmethod
    def _prep_protalbert(seqs):
        cleaned = []
        for s in seqs:
            s = s.upper().replace("U", "X").replace("O", "X")
            cleaned.append(" ".join(list(s)))
        return cleaned

    def predict(self, batch, all_loss=None, metric=None):

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        graphs = batch["graph"]
        self.model.to(device).eval() 
        self.mlp.to(device).eval()


        # Convert graphs -> sequences
        sequences = []
        for graph in graphs:
            residue_ids = graph.residue_type.tolist()
            seq = "".join(data.Protein.id2residue_symbol[res] for res in residue_ids)
            sequences.append(seq)

        batch_size = len(sequences)

        # We'll store final chosen logits + chosen layer
        final_logits = [None] * batch_size
        final_layers = [None] * batch_size
        best_prob     = torch.full((batch_size,), -float("inf"), device=device)   # NEW
        best_logits   = [None] * batch_size                                       # NEW
        best_layers   = [None] * batch_size      
        computed_layers = [None] * batch_size
        active = torch.arange(batch_size, device=device)                                 

        # 1) Tokenize once
        prepared = self._prep_protalbert(sequences)                       # EDIT
        enc = self.tokenizer(
            prepared,
            add_special_tokens=True,
            padding=True,
            truncation=True,          # ProtAlbert max = 512
            return_tensors="pt",
            max_length = 550,
        )
        input_ids      = enc["input_ids"     ].to(device)
        attention_mask = enc["attention_mask"].to(device)


        # 5) Temperatures
        threshold = float(os.getenv("THRESHOLD"))
        # temperature_file = os.getenv("TEMPERATURE_FILE")
        # if temperature_file is not None and temperature_file != 'None':
        #     temperatures = self.extract_temperatures(temperature_file)
        #     temperatures = torch.tensor(temperatures, device=device)
        # else:
        n_layers  = self.model.config.num_hidden_layers
        temps = torch.ones(n_layers, device=device)
        

        hs = self.model.embeddings(input_ids)
        hs = self.model.encoder.embedding_hidden_mapping_in(hs)
        attn_ext = (1.0 - attention_mask[:, None, None, :]) * -10000.0

        groups = self.model.encoder.albert_layer_groups
        layers_per_group = n_layers // self.model.config.num_hidden_groups
        global_layer_idx = 0

        for group in groups:
            for _ in range(layers_per_group):
                if len(active) == 0:
                    break
                for idx in active.tolist():
                    computed_layers[idx] = global_layer_idx

                # ---- forward one logical layer ----
                hs_active = hs[active]                                                # (N,L,H)
                hs_active = group(
                    hs_active,
                    attention_mask=attn_ext[active],
                    head_mask=[None] * self.model.config.num_hidden_layers,  # ← FIX HERE
                    output_attentions=False,
                )
                hs_active = hs_active[0] if isinstance(hs_active, tuple) else hs_active
                hs[active] = hs_active

                # ---- classifier ----
                pooled = hs_active.mean(dim=1)                                        # (N,H)
                logits = self.mlp[global_layer_idx](pooled)
                prob   = torch.sigmoid(logits / temps[global_layer_idx])
                max_p, _ = prob.max(dim=1)

                # ---- best‑so‑far bookkeeping ----
                is_final = global_layer_idx == n_layers - 1
                better   = max_p > best_prob[active]
                if is_final and os.getenv("SELECT_LAST", "False") == "True":
                    better = torch.ones_like(better, dtype=torch.bool)

                if better.any():
                    g_idx = active[better]
                    best_prob[g_idx] = max_p[better]
                    for j, gi in enumerate(g_idx.tolist()):
                        best_logits[gi] = logits[better][j]
                        best_layers[gi] = global_layer_idx

                # ---- early‑exit decision ----
                exit_mask   = max_p > threshold
                newly_exit  = active[exit_mask]
                still_act   = active[~exit_mask]

                for j, gi in enumerate(newly_exit.tolist()):
                    final_logits[gi] = logits[exit_mask][j]
                    final_layers[gi] = global_layer_idx

                active = still_act
                global_layer_idx += 1
            if len(active) == 0:
                break

        # ---------- FORCE EXIT REMAINDERS ----------
        for gi in active.tolist():
            final_logits[gi] = best_logits[gi]
            final_layers[gi] = best_layers[gi]

        # ---------- STACK & RETURN ----------
        preds = torch.stack(final_logits, dim=0)
        # keep legacy 2000‑wide ASCII tensor (unchanged)
        ascii_mat = torch.nn.utils.rnn.pad_sequence(
            [torch.tensor([ord(c) for c in s], device=device) for s in sequences],
            batch_first=True, padding_value=0,
        )
        if ascii_mat.size(1) < 2000:
            pad = ascii_mat.new_zeros(ascii_mat.size(0), 2000 - ascii_mat.size(1))
            ascii_mat = torch.cat([ascii_mat, pad], dim=1)

        return {
            "pred": preds,
            "layers": torch.tensor(final_layers, device=device, dtype=torch.int64),
            "computed_layers":torch.tensor(computed_layers, device=self.device, dtype=torch.int64),
            "sequences": ascii_mat,
        }

    def target(self, batch):
        return batch["targets"]

    def evaluate(self, preds, target):
        result = {}
        pred  = preds["pred"]
        layers = preds["layers"]
        computed_layers = preds["computed_layers"]
        sequences = preds["sequences"]
        target = target.to(pred.device)

        metric_out = {}
        m = self.metric
        if m == "f1_max":
            score = metrics.f1_max(pred, target)
            metric_out[m] = score.item()
        else:
            raise ValueError(f"Unknown metric {m}")

        freq = torch.bincount(layers.cpu())
        avg_layer = (torch.arange(len(freq), device=freq.device) * freq).sum() / freq.sum()

        computed_layer_frequencies = torch.bincount(computed_layers)
        total_computed = computed_layer_frequencies.sum()
        computed_layer_indices = torch.arange(len(computed_layer_frequencies), device=computed_layer_frequencies.device)
        average_computed_layer = (computed_layer_indices * computed_layer_frequencies).sum() / total_computed


        result["avg_layer"] = avg_layer.item()
        result["avg_computed_layer"] = average_computed_layer.item()
        result[m] = metric_out[m]
        return result
    


##### CONFIDENCE METRICS #######
@R.register("tasks.Classification_confidence_ProtBert")
class Classification_confidence_ProtBert(Classification_walltime_ProtBert):
    def __init__(self, model, metric=('f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer):
        super().__init__(model, metric=('f1_max'), verbose=0, num_class=1, weight=None, tokenizer=AutoTokenizer)
    def evaluate(self, preds, target):
        metric_out = evaluate_classification(preds, target, self.metric)
        return metric_out

@R.register("tasks.Classification_confidence_ESM")
class Classification_confidence_ESM(Classification_walltime_ESM):
    def __init__(self, model, metric=('f1_max'), verbose=0, num_class=1, weight=None, confidence_threshold=None):
        super().__init__(model, metric=('f1_max'), verbose=0, num_class=1, weight=None, confidence_threshold=None)
    def evaluate(self, preds, target):
        metric_out = evaluate_classification(preds, target, self.metric, self.num_class)
        return metric_out
  
@R.register("tasks.Property_confidence_ProtBert") 
class Property_confidence_ProtBert(Property_walltime_ProtBert):
    def __init__(self, model, task=(), metric=("acc"), criterion="mse", num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super().__init__(model, task=task, metric=metric, criterion=criterion, num_mlp_layer=num_mlp_layer, #switched to 2
                 normalization=False, num_class=num_class, mlp_batch_norm = mlp_batch_norm, mlp_dropout = mlp_dropout,
                 graph_construction_model=None, confidence_threshold = None, verbose=0)
    def evaluate(self, preds, target):
        metric_out = evaluate_property_confidence(preds, target, self.num_class)
        return metric_out
    

@R.register("tasks.Property_confidence_ESM")
class Property_confidence_ESM(NormalProperty_continuous):
    def __init__(self, model, task=(), criterion="mse", metric=("mae", "rmse"), num_mlp_layer=2, #switched to 2
                 normalization=False, num_class=None, mlp_batch_norm=False, mlp_dropout=0,
                 graph_construction_model=None, confidence_threshold = None, verbose=0):
        super().__init__(model, task=task, metric=metric, criterion=criterion, num_mlp_layer=num_mlp_layer, #switched to 2
                 normalization=False, num_class=num_class, mlp_batch_norm=mlp_batch_norm, mlp_dropout=mlp_dropout,
                 graph_construction_model=None, confidence_threshold = None, verbose=0)
    def evaluate(self, preds, target):
        metric_out = evaluate_property_confidence(preds, target, self.num_class)
        return metric_out
    

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


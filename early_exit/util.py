import os
import time
import logging
import argparse

import yaml
import jinja2
from jinja2 import meta
import easydict

import torch
from torch import distributed as dist
from torch.optim import lr_scheduler

from torchdrug import core, utils, datasets, models, tasks, layers
from torchdrug.utils import comm
from torch import nn


logger = logging.getLogger(__file__)


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

def get_root_logger(file=True):
    logger = logging.getLogger("")
    logger.setLevel(logging.INFO)
    format = logging.Formatter("%(asctime)-10s %(message)s", "%H:%M:%S")

    if file:
        handler = logging.FileHandler("log.txt")
        handler.setFormatter(format)
        logger.addHandler(handler)

    return logger


def create_working_directory(cfg):
    file_name = "working_dir.tmp"
    world_size = comm.get_world_size()
    if world_size > 1 and not dist.is_initialized():
        comm.init_process_group("nccl", init_method="env://")

    working_dir = os.path.join(os.path.expanduser(cfg.output_dir),
                               cfg.task["class"], cfg.dataset["class"] + cfg.dataset.get("level", ""), cfg.task.get("model", cfg.get("p_model"))["class"],
                               time.strftime("%Y-%m-%d-%H-%M-%S"))

    # synchronize working directory
    if comm.get_rank() == 0:
        with open(file_name, "w") as fout:
            fout.write(working_dir)
        os.makedirs(working_dir)
    comm.synchronize()
    if comm.get_rank() != 0:
        with open(file_name, "r") as fin:
            working_dir = fin.read()
    comm.synchronize()
    if comm.get_rank() == 0:
        os.remove(file_name)

    os.chdir(working_dir)
    return working_dir


def detect_variables(cfg_file):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    env = jinja2.Environment()
    ast = env.parse(raw)
    vars = meta.find_undeclared_variables(ast)
    return vars


def load_config(cfg_file, context=None):
    with open(cfg_file, "r") as fin:
        raw = fin.read()
    template = jinja2.Template(raw)
    instance = template.render(context)
    cfg = yaml.safe_load(instance)
    cfg = easydict.EasyDict(cfg)
    return cfg


def build_downstream_solver(cfg, dataset):
    if "test_split" in cfg:
        train_set, valid_set, test_set = dataset.split(['train', 'valid', cfg.test_split])
    elif isinstance(dataset, tuple):
        train_set, valid_set, test_set = dataset
        num_classes = train_set.num_classes
        weights = getattr(train_set, "weights", torch.ones((train_set.num_classes,), dtype=torch.float))
    else:
        train_set, valid_set, test_set = dataset.split()
        num_classes = len(dataset.targets)
        weights = getattr(dataset, "weights", torch.ones((len(dataset.targets),), dtype=torch.float))
    if comm.get_rank() == 0:
        logger.warning(dataset)
        logger.warning("#train: %d, #valid: %d, #test: %d" % (len(train_set), len(valid_set), len(test_set)))

    if cfg.task['class'] == "FunctionAnnotation":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "FoldClassification":
        cfg.task.num_class = num_classes
    elif cfg.task['class'] == "FunctionAnnotation_AllLayers":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "EarlyExitClassificationTemperature_Calibration":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "EarlyExitClassificationTemperature":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "EarlyExitClassificationTemperature_Analysis":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "EarlyExitClassification_walltime_ProtAlbert":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "EarlyExitClassification_walltime_ProtBert":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
    elif cfg.task['class'] == "NodePropertyPredictionAllLayers": #added
        cfg.task.num_class = 3
    elif cfg.task['class'] in ["EarlyExitClassificationTemperature_Node", "EarlyExitClassificationTemperature_Node_continuous", "ClassificationTemperature_Node_continuous", "EarlyExitClassificationTemperature_Node_continuous_ProtBert", "EarlyExitClassificationTemperature_Node_continuous_ProtAlbert", "ClassificationTemperature_Node_continuous_ProtBert", "ClassificationTemperature_Node_continuous_ProtAlbert"]:
        cfg.task.num_class = 3
    elif cfg.task['class'] in ["EarlyExitClassification_walltime", "EarlyExitClassification_walltime_analysis", "Classification_walltime_ProtBert", "Classification_walltime_ProtAlbert"]:
        cfg.task.num_class = num_classes
        cfg.task.weight = weights

    elif cfg.task['class'] == "NormalClassification_walltime":
        cfg.task.num_class = num_classes
        cfg.task.weight = weights
        #cfg.task.weight = weights
    else:
        cfg.task.task = dataset.tasks
    task = core.Configurable.load_config_dict(cfg.task)

    cfg.optimizer.params = task.parameters()        
    optimizer = core.Configurable.load_config_dict(cfg.optimizer)

    if "scheduler" not in cfg:
        scheduler = None
    elif cfg.scheduler["class"] == "ReduceLROnPlateau":
        cfg.scheduler.pop("class")
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    else:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        cfg.engine.scheduler = scheduler

    solver = core.Engine(task, train_set, valid_set, test_set, optimizer, **cfg.engine)

    if "lr_ratio" in cfg:
        if cfg.lr_ratio > 0:
            cfg.optimizer.params = [
                {'params': solver.model.model.parameters(), 'lr': cfg.optimizer.lr * cfg.lr_ratio},
                {'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}
            ]
        else:
            for p in solver.model.model.parameters():
                p.requires_grad = False
            cfg.optimizer.params = [{'params': solver.model.mlp.parameters(), 'lr': cfg.optimizer.lr}]
        optimizer = core.Configurable.load_config_dict(cfg.optimizer)
        solver.optimizer = optimizer

    if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, **cfg.scheduler)
    elif scheduler is not None:
        cfg.scheduler.optimizer = optimizer
        scheduler = core.Configurable.load_config_dict(cfg.scheduler)
        solver.scheduler = scheduler


    if cfg.get("pretrained_mlp_load") is not None:
        print("warning: getting pretrained mlp load")
        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            checkpoint = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
            print(f"Getting model keys")
            print(checkpoint["model"].keys())
            if cfg.load_weights_only:  # Load model weights
                model_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint["model"].items() if k.startswith('model.model.')}
                task.model.load_state_dict(model_state_dict, strict=True)
                # Check if task has MLP layers and initialize if missing
                if not hasattr(task, 'mlp'):
                    # Define your MLP layers here based on expected configurations, e.g.,
                    num_layers = task.model.num_layers  # Assuming model has num_layers defined
                    print(f"num_layers {num_layers}")
                    print(f"output_dim {task.model.output_dim}")
                    print(f"num_class {cfg.task.num_class}")
                    task.mlp = nn.ModuleList([
                        MLP(in_channels=task.model.output_dim, mid_channels=task.model.output_dim, 
                            out_channels=cfg.task.num_class, batch_norm=True, dropout=0.2) #will need to edit if configuration of custom_esm2_ecallLayers changes 
                        for _ in range(num_layers)
                    ]).to(task.model.device)

                # Load the MLP layers from the checkpointq
                mlp_state_dict = {k.replace('mlp.', '', 1): v for k, v in checkpoint["model"].items() if k.startswith('mlp.')}
                task.mlp.load_state_dict(mlp_state_dict, strict=True)
                for param in task.mlp.parameters(): #AH 11/18
                    param.requires_grad = False

            else:  # Load complete model with the entire state_dict
                model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
                task.model.load_state_dict(model_dict)
    elif cfg.get("pretrained_mlp_load_albert"):
        print("warning: getting pretrained mlp load albert")
        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            checkpoint = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
            print(f"Getting model keys")
            print(checkpoint["model"].keys())
            if cfg.load_weights_only:  # Load model weights
                from transformers import AutoModel, AlbertTokenizer
                task.tokenizer = AlbertTokenizer.from_pretrained("Rostlab/prot_albert", cache_dir="/scratch/anna19/cache2/", use_fast=False, do_lower_case=False, do_basic_tokens=False)
                task.model = AutoModel.from_pretrained("Rostlab/prot_albert", cache_dir="/scratch/anna19/cache2/")
                task.model.eval()
                for param in task.model.parameters():
                    param.requires_grad = False
                task.model.num_layers = task.model.config.num_hidden_layers 
                # Check if task has MLP layers and initialize if missing
                if not hasattr(task, 'mlp'):
                    # Define your MLP layers here based on expected configurations, e.g.,
                    num_layers = task.model.num_layers  # Assuming model has num_layers defined
                    print(f"num_layers {num_layers}")
                    print(f"output_dim {task.model.config.hidden_size}")
                    print(f"num_class {cfg.task.num_class}")
                    task.mlp = nn.ModuleList([
                        MLP(in_channels=task.model.config.hidden_size, mid_channels=task.model.config.hidden_size, 
                            out_channels=cfg.task.num_class, batch_norm=True, dropout=0.2) #will need to edit if configuration of custom_esm2_ecallLayers changes 
                        for _ in range(num_layers)
                    ]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # Load the MLP layers from the checkpointq
                mlp_state_dict = {k.replace('mlp.', '', 1): v for k, v in checkpoint["model"].items() if k.startswith('mlp.')}
                task.mlp.load_state_dict(mlp_state_dict, strict=True)
                for param in task.mlp.parameters(): #AH 11/18
                    param.requires_grad = False
                task.mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            else:  # Load complete model with the entire state_dict
                model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
                task.model.load_state_dict(model_dict)
    elif cfg.get("pretrained_mlp_load_bert"):
        print("warning: getting pretrained mlp load bert")
        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            checkpoint = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
            print(f"Getting model keys")
            print(checkpoint["model"].keys())
            for k, v in checkpoint["model"].items():
                if "mlp" in k and "weight" in k and v.dim() == 2:
                    print(f"{k}: {v.shape}")
            if cfg.load_weights_only:  # Load model weights
                from transformers import AutoModel, AutoTokenizer
                task.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir="/scratch/anna19/cache2/", )
                task.model = AutoModel.from_pretrained("Rostlab/prot_bert", cache_dir="/scratch/anna19/cache2/")
                task.model.eval()
                for param in task.model.parameters():
                    param.requires_grad = False
                task.model.num_layers = task.model.config.num_hidden_layers 
                # Check if task has MLP layers and initialize if missing
                if not hasattr(task, 'mlp'):
                    # Define your MLP layers here based on expected configurations, e.g.,
                    num_layers = task.model.num_layers  # Assuming model has num_layers defined
                    print(f"num_layers {num_layers}")
                    print(f"output_dim {task.model.config.hidden_size}")
                    print(f"num_class {cfg.task.num_class}")
                    task.mlp = nn.ModuleList([
                        MLP(in_channels=task.model.config.hidden_size, mid_channels=task.model.config.hidden_size, 
                            out_channels=cfg.task.num_class, batch_norm=True, dropout=0.2) #will need to edit if configuration of custom_esm2_ecallLayers changes 
                        for _ in range(num_layers)
                    ]).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # Load the MLP layers from the checkpointq
                mlp_state_dict = {k.replace('mlp.', '', 1): v for k, v in checkpoint["model"].items() if k.startswith('mlp.')}
                task.mlp.load_state_dict(mlp_state_dict, strict=True)
                for param in task.mlp.parameters(): #AH 11/18
                    param.requires_grad = False
                task.mlp.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

            else:  # Load complete model with the entire state_dict
                model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
                task.model.load_state_dict(model_dict)
    elif cfg.get("pretrained_mlp_load_property"):
        print("warning: getting pretrained mlp load: PROPERTY")
        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            checkpoint = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
            print(f"Getting model keys")
            print(checkpoint["model"].keys())
            if cfg.load_weights_only:  # Load model weights
                model_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint["model"].items() if k.startswith('model.model.')}
                task.model.load_state_dict(model_state_dict, strict=True)
                # Check if task has MLP layers and initialize if missing
                if not hasattr(task, 'mlp'):
                    # Define your MLP layers here based on expected configurations, e.g.,
                    num_layers = task.model.num_layers  # Assuming model has num_layers defined
                    print(f"num_layers {num_layers}")
                    print(f"output_dim {task.model.output_dim}")
                    print(f"num_class {cfg.task.num_class}")
                    num_mlp_layer = cfg.task.num_mlp_layer
                    hidden_dims = [task.model.output_dim] * (num_mlp_layer - 1)
                    task.mlp = nn.ModuleList([layers.MLP(task.model.output_dim, hidden_dims + [cfg.task.num_class],
                            batch_norm=False, dropout=0)
                            for _ in range(num_layers)]).to(task.model.device)
                    # task.mlp = nn.ModuleList([
                    #     MLP(in_channels=task.model.output_dim, mid_channels=task.model.output_dim, 
                    #         out_channels=cfg.task.num_class, batch_norm=True, dropout=0.2) #will need to edit if configuration of custom_esm2_ecallLayers changes 
                    #     for _ in range(num_layers)
                    # ]).to(task.model.device)

                # Load the MLP layers from the checkpointq
                mlp_state_dict = {k.replace('mlp.', '', 1): v for k, v in checkpoint["model"].items() if k.startswith('mlp.')}
                task.mlp.load_state_dict(mlp_state_dict, strict=True)
                for param in task.mlp.parameters(): #AH 11/18
                    param.requires_grad = False

            else:  # Load complete model with the entire state_dict
                model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
                task.model.load_state_dict(model_dict)

    elif cfg.get("pretrained_mlp_load_albert_property"):
        print("warning: getting pretrained mlp load: ALBERT‑PROPERTY")
        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            checkpoint = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
            print(f"Getting model keys")
            print(checkpoint["model"].keys())

            if cfg.load_weights_only:  # Load only the MLP weights
                # ----- 1.  Build a *frozen* ProtALBERT backbone -----
                from transformers import AutoModel, AlbertTokenizer
                task.tokenizer = AlbertTokenizer.from_pretrained(
                    "Rostlab/prot_albert",
                    cache_dir="/scratch/anna19/cache2/",
                    use_fast=False,
                    do_lower_case=False,
                    do_basic_tokens=False,
                )
                task.model = AutoModel.from_pretrained(
                    "Rostlab/prot_albert",
                    cache_dir="/scratch/anna19/cache2/",
                )
                task.model.eval()
                for param in task.model.parameters():
                    param.requires_grad = False

                # Hugging Face ALBERT has `config.num_hidden_layers`
                task.model.num_layers = task.model.config.num_hidden_layers
                hidden_size = task.model.config.hidden_size

                # ----- 2.  Build per‑layer MLP heads (PROPERTY style) -----
                if not hasattr(task, "mlp"):
                    num_layers = task.model.num_layers
                    num_mlp_layer = cfg.task.num_mlp_layer        # e.g. 2
                    hidden_dims = [hidden_size] * (num_mlp_layer - 1)

                    task.mlp = nn.ModuleList(
                        [
                            layers.MLP(
                                hidden_size,
                                hidden_dims + [cfg.task.num_class],
                                batch_norm=False,
                                dropout=0,
                            )
                            for _ in range(num_layers)
                        ]
                    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # ----- 3.  Load the saved MLP parameters -----
                mlp_state_dict = {
                    k.replace("mlp.", "", 1): v
                    for k, v in checkpoint["model"].items()
                    if k.startswith("mlp.")
                }
                task.mlp.load_state_dict(mlp_state_dict, strict=True)

                # Freeze the MLPs so they are inference‑only
                for param in task.mlp.parameters():
                    param.requires_grad = False

            else:  # Load the whole checkpoint (model + MLP) at once
                model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
                task.model.load_state_dict(model_dict)

    elif cfg.get("pretrained_mlp_load_property_bert"):
        print("warning: getting pretrained mlp load: BERT‑PROPERTY")
        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            checkpoint = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
            print("Getting model keys")
            print(checkpoint["model"].keys())

            if cfg.load_weights_only:  # ----- load only the MLP heads -----
                # 1) Build a *frozen* ProtBERT backbone
                from transformers import AutoModel, AutoTokenizer
                task.tokenizer = AutoTokenizer.from_pretrained(
                    "Rostlab/prot_bert",
                    cache_dir="/scratch/anna19/cache2/",
                    do_lower_case=False,
                )
                task.model = AutoModel.from_pretrained(
                    "Rostlab/prot_bert",
                    cache_dir="/scratch/anna19/cache2/",
                )
                task.model.eval()
                for p in task.model.parameters():
                    p.requires_grad = False

                task.model.num_layers = task.model.config.num_hidden_layers
                hidden_size = task.model.config.hidden_size

                # 2) Build per‑layer MLP heads (PROPERTY style)
                if not hasattr(task, "mlp"):
                    num_layers     = task.model.num_layers
                    num_mlp_layer  = cfg.task.num_mlp_layer        # e.g. 2
                    hidden_dims    = [hidden_size] * (num_mlp_layer - 1)

                    task.mlp = nn.ModuleList(
                        [
                            layers.MLP(
                                hidden_size,
                                hidden_dims + [cfg.task.num_class],
                                batch_norm=False,
                                dropout=0,
                            )
                            for _ in range(num_layers)
                        ]
                    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

                # 3) Load the saved MLP weights
                mlp_state_dict = {
                    k.replace("mlp.", "", 1): v
                    for k, v in checkpoint["model"].items()
                    if k.startswith("mlp.")
                }
                task.mlp.load_state_dict(mlp_state_dict, strict=True)

                # Freeze the MLPs (inference‑only)
                for p in task.mlp.parameters():
                    p.requires_grad = False

            else:  # ----- load entire checkpoint (backbone + MLP heads) -----
                model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device("cpu"))
                task.model.load_state_dict(model_dict)
    else:
        if cfg.get("checkpoint") is not None:
            solver.load(cfg.checkpoint)

        if cfg.get("model_checkpoint") is not None:
            if comm.get_rank() == 0:
                logger.warning("Load checkpoint from %s" % cfg.model_checkpoint)
            cfg.model_checkpoint = os.path.expanduser(cfg.model_checkpoint)
            model_dict = torch.load(cfg.model_checkpoint, map_location=torch.device('cpu'))
            task.model.load_state_dict(model_dict)


    
    return solver, scheduler


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", help="yaml configuration file", required=True)
    parser.add_argument("-s", "--seed", help="random seed for PyTorch", type=int, default=1024)

    args, unparsed = parser.parse_known_args()
    # get dynamic arguments defined in the config file
    vars = detect_variables(args.config)
    parser = argparse.ArgumentParser()
    for var in vars:
        parser.add_argument("--%s" % var, default="null")
    vars = parser.parse_known_args(unparsed)[0]
    vars = {k: utils.literal_eval(v) for k, v in vars._get_kwargs()}

    return args, vars
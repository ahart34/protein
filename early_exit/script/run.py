import os
import sys
import math
import pprint
import random

import numpy as np

import torch
from torch.optim import lr_scheduler
import os
import pickle
import time
from torchdrug import core, models, tasks, datasets, utils
from torchdrug.utils import comm
import sys
sys.path.append('/shared/nas2/anna19/protein/early_exit/model')
from custom_esm2 import CustomESM2
from custom_protbert import CustomProtBert
from custom_protalbert import CustomProtAlbert
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import util
import dataset, task
import glob
import csv
import pdb
import json


def train_and_validate_all_layers(cfg, solver, scheduler):
    print("training and validating all layers")
    if cfg.train.num_epoch == 0:
        return
    checkpoint_path = cfg.checkpoint_path
    os.makedirs(checkpoint_path, exist_ok=True)
    step = math.ceil(cfg.train.num_epoch / 500)
    best_result = float("-inf")
    best_epoch = -1

    for i in range(0, cfg.train.num_epoch, step):
        print(f"epoch {i}")
        kwargs = cfg.train.copy()
        kwargs["num_epoch"] = min(step, cfg.train.num_epoch - i)
        solver.train(**kwargs)
        # first = solver.model.mlp[0].module[-1]
        # print(f"[EPOCH {i}] Linear in/out = {first.in_features}/{first.out_features}")
        print("finish train")
        metric = solver.evaluate("valid")
        for k, v in metric.items():
           if k.startswith(cfg.eval_metric):
               result = v
               print(f"{k}: {result}")

        if not math.isnan(result.item()) and result.item() > best_result:
            best_result = result.item()
            best_epoch = i
            best_epoch = solver.epoch
            checkpoint_file = os.path.join(checkpoint_path, "model_epoch%d.pth" % solver.epoch)
            for fn in glob.glob(checkpoint_path):
                try:
                    if os.path.isfile(fn):
                        os.remove(fn)
                except FileNotFoundError:
                    pass
            solver.save(checkpoint_file)
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(result)
        print("scheduler stepped")
    print(f"best epoch = {best_epoch}")
    best_checkpoint = os.path.join(checkpoint_path, "model_epoch%d.pth" % best_epoch)
    solver.load(best_checkpoint)
    evaluate_by_layers(cfg, solver, scheduler)
    print(f"best_epoch = {best_epoch}")
    #return solver
    return

def parse_to_csv(data, fn):
    import re
    import csv
    pattern = re.compile(r"^(.+?) Layer (\d+)$")             #flipped to pattern = re.compile(r"^layer (\d+) (.+)$")  for EC, GO               
    parsed_data = {}
    for key, value in data.items():
        match = pattern.match(key)
        if match:
            layer_idx = int(match.group(2))
            metric_name = match.group(1)
            if layer_idx not in parsed_data:
                parsed_data[layer_idx] = {}
            parsed_data[layer_idx][metric_name] = value
    all_metrics = sorted({metric for layer in parsed_data.values() for metric in layer.keys()})
    with open(fn, 'w', newline = '') as f:
        writer = csv.writer(f)
        writer.writerow(["Layer"] + all_metrics)
        for layer_idx in sorted(parsed_data.keys()):
            row = [layer_idx] + [parsed_data[layer_idx].get(metric, "").item() for metric in all_metrics]
            writer.writerow(row)
    return

def evaluate_by_layers(cfg, solver, scheduler):
    #evaluation_results = solver.evaluate("valid")
    #parse_to_csv(evaluation_results, cfg.evaluation_path_valid)
    evaluation_results = solver.evaluate("test")
    parse_to_csv(evaluation_results, cfg.evaluation_path_test)
    return 

def evaluate_multiple_thresholds(cfg, solver, scheduler, last=False):
    os.environ["RESULT_FILE"] = "/shared/nas2/anna19/early_exit/esm-s/results/non_necessary"
    import csv
    if last== True:
        os.environ["SELECT_LAST"] = "True"
        print("selecting last")
    result_file = cfg.get("evaluation_result_file_stem")
    os.makedirs(cfg.result_pickle, exist_ok=True)
    #from fvcore.nn import FlopCountAnalysis
    # def wrapped_predict(batch, *args, **kwargs):
    #     nonlocal total_flops_this_threshold
    #     # count flops on this batch
    #     tensor_input = batch["graph"]
    #     batch_flops = FlopCountAnalysis(solver.model, tensor_input).total()
    #     total_flops_this_threshold += batch_flops
    #     return original_predict(batch, *args, **kwargs)
    #for t in np.arange(0, 1, .01):
    #for t in np.arange(0, 1, .01):
    if cfg.get("property") is not None:
        print("evaluating property")
        evaluation_result_file = f"{result_file}.csv" #Insert evaluation_result_file here 
        #for t in np.concatenate([np.arange(0, 1, .04), np.arange(.982, 1, .002), np.arange(.9982, 1, .0002), np.array([1.0])]): #for esm2
        for t in np.concatenate([np.arange(0, .8, .2), np.arange(.8, 1, .08), np.arange(.982, 1, .004), np.arange(.999, 1, .0002), np.array([1.0])]): #for albert
        #for t in np.concatenate([np.arange(0.98, 1, 0.001), np.arange(0.999, 1, 0.0001), np.arange(.9999, 1, .00001), np.array([1.0])]):
        #for t in np.arange(.999, 1, .0001):
        #for t in np.arange(.9999, 1, .00001):
        #for t in np.concatenate([np.arange(0, .88, .08), np.arange(.88, .96, .02), np.arange(.96, 1, .01), np.arange(.99, 1, .005), np.array([1.0])]):
            #total_flops_this_threshold = 0
            #original_predict = solver.model.predict
            #solver.model.predict = wrapped_predict
            os.environ["THRESHOLD"] = str(t)
            os.environ["RESULT_PICKLE"] = f"{cfg.result_pickle}/run_{t:.5f}.pkl"
            print(f"evaluating on threshold {t}")
            times = time.time()
            metric = solver.evaluate("test")
            timef = time.time()
            acc = metric["acc"]
            timee = timef - times
            layer = metric["layer"]
            with open(evaluation_result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([t, acc, layer, timee])

    else:
        #for percent in [30, 50,95]:
        for percent in [95]:
            cfg.dataset.split = "test"
            print(f"evaluating percent {percent}")
            cfg.dataset.percent = percent
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            solver.test_set = test_set
            evaluation_result_file = f"{result_file}_percent{percent}.csv" #Insert evaluation_result_file here 
            os.environ["RESULT_FILE"] = str(evaluation_result_file)
            #for t in np.concatenate([np.arange(0, 1, .02), np.arange(.982, 1, .002), np.arange(.9982, 1, .0002), np.array([1.0])]):
            #for t in np.concatenate([np.arange(0.98, 1, 0.001), np.arange(0.999, 1, 0.0001), np.arange(.9999, 1, .00001), np.array([1.0])]):
            #for t in np.arange(.999, 1, .0001):
            #for t in np.arange(.9999, 1, .00001):
            
            for t in np.concatenate([np.arange(0, .88, .08), np.arange(.88, .96, .02), np.arange(.96, 1, .01), np.arange(.99, 1, .005), np.array([1.0])]):
                os.environ["THRESHOLD"] = str(t)
                os.environ["RESULT_PICKLE"] = f"{cfg.result_pickle}/run_{percent}_{t:.5f}.pkl"
                print(f"evaluating on threshold {t}")
                cfg.threshold = t
                time_s = time.time()
                result = solver.evaluate("test")
                time_f = time.time()
                timee = time_f - time_s
                with open(evaluation_result_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([os.getenv("THRESHOLD"), result["f1"], result["avg_layer"], timee])

    return None

if __name__ == "__main__":
    args, vars = util.parse_args()
    cfg = util.load_config(args.config, context=vars)
    working_dir = util.create_working_directory(cfg)
    
    seed = args.seed
    print(f"SEED = {seed}")
    torch.manual_seed(seed + comm.get_rank())
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger = util.get_root_logger()
    if comm.get_rank() == 0:
        logger.warning("Config file: %s" % args.config)
        logger.warning(pprint.pformat(cfg))

    if cfg.get("evaluate_layer_only"): #AH
        ##CODE FOR EVALUATING PLAIN MODEL WITH MLP AT EACH LAYER# 
        print("evalute by layers")
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (valid_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)
        evaluate_by_layers(cfg, solver, scheduler)
    elif cfg.get("train_all_layers"):
        print("training all layers")
        #CODE FOR TRAINING ALL MLPS IN LAYERS##
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (train_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)
        train_and_validate_all_layers(cfg, solver, scheduler)
    elif cfg.get("evaluate"):
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (train_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)
        with open(cfg.get("result_file"), 'w') as f:
            writer = csv.writer(f)
            writer.writerow(["Layer", cfg.task.metric, "Time"])
            for layer in range(0, int(cfg.get("num_layers"))):
                os.environ["LAYER"] = str(layer)
                times = time.time()
                results = solver.evaluate("test")   
                timef = time.time()   
                timee = timef - times
                metric = results[cfg.task.metric]  
                writer.writerow([layer, metric, timee])
                f.flush()
    elif cfg.get("evaluate_confidence"):
        os.environ["CONFIDENCE_CALIBRATION"] = "True"
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (train_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)
        num_layers = int(cfg.get("num_layers"))
        all_layer_preds = []
        target = None
        os.environ["OUT_FILE"] = str(cfg.out_file)
        for layer in range(num_layers):
            os.environ["LAYER"] = str(layer)
            results = solver.evaluate("test")  # should return dict with preds/target/etc.
            
        #     # Grab target once (assumes target doesn't depend on layer)
        #     if target is None:
        #         target = results["target"]
        #         # Ensure target is a tensor on CPU for saving
        #         if torch.is_tensor(target):
        #             target = target.detach().cpu()
        #         else:
        #             target = torch.as_tensor(target)

        #     preds = results["preds"]

        #     # Normalize preds into a single tensor [n_examples, n_classes] for this layer
        #     # Common cases:
        #     #  - preds is already a tensor
        #     #  - preds is a list of tensors (batches) -> cat
        #     if isinstance(preds, (list, tuple)):
        #         preds = torch.cat([p.detach().cpu() if torch.is_tensor(p) else torch.as_tensor(p)
        #                         for p in preds], dim=0)
        #     else:
        #         preds = preds.detach().cpu() if torch.is_tensor(preds) else torch.as_tensor(preds)

        #     all_layer_preds.append(preds)

        # # Stack into [n_layers, n_examples, n_classes]
        # preds_by_layer = torch.stack(all_layer_preds, dim=0)

        # out_file_base = cfg.get("out_file_base")
        # pt_path = out_file_base + ".pt"
        # meta_path = out_file_base + ".json"

        # payload = {
        #     "preds_by_layer": preds_by_layer,  # [L, N, C]
        #     "target": target               # [N, C] (multi-label) or [N] / [N,1] depending on task
        # }
        # torch.save(payload, pt_path)

        # # Metadata (don’t assume target is always 2D)
        # n_examples = int(target.shape[0]) if hasattr(target, "shape") else None
        # n_classes = int(target.shape[1]) if (hasattr(target, "shape") and len(target.shape) > 1) else None

        # with open(meta_path, "w") as f:
        #     json.dump(
        #         {
        #             "dataset_class": cfg.dataset["class"] if isinstance(cfg.dataset, dict) else str(cfg.dataset),
        #             "n_examples": n_examples,
        #             "n_classes": n_classes,
        #             "n_layers": num_layers,
        #             "preds_shape": list(preds_by_layer.shape),
        #             "target_shape": list(target.shape) if hasattr(target, "shape") else None,
        #             "task_metric": str(cfg.task.metric),
        #         },
        #         f,
        #         indent=2,
        #     )
    elif cfg.get("exit"):
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (valid_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)

        for select_last in ("True", "False"):
            os.environ["SELECT_LAST"] = select_last
            result_file = cfg.get("result_file")
            root, ext = os.path.splitext(result_file)
            if select_last == "True":
                result_file = f"{root}_last{ext}"
            else:
                result_file = f"{root}_max{ext}"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            solver.test_set = test_set
            avg_layer = 0
            d = 1

            ##Getting max layer ### 
            t = 1 
            os.environ["THRESHOLD"] = str(t)
            print(f"evaluating on threshold {t}")
            times = time.time()
            metric = solver.evaluate("test")
            timef = time.time()
            acc = metric[cfg.task.metric]
            timee = timef - times
            layer = metric["avg_layer"]
            computed_layer = metric["avg_computed_layer"]
            with open(result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["Threshold", cfg.task.metric, "Average Exit Layer", "Average Computed Layer", "Time"])
                writer.writerow([t, acc, layer, computed_layer, timee])
            max_layer = layer
            t = 0
            layer = 0
            while layer < max_layer * .95: 
                target = 1 - 2*10**(-d)
                step = 2*10**(-d)
                while t < target: 
                    os.environ["THRESHOLD"] = str(t) 
                    times = time.time()
                    metric = solver.evaluate("test")
                    timef = time.time()
                    acc = metric[cfg.task.metric]
                    timee = timef - times
                    layer = metric["avg_layer"]
                    computed_layer = metric["avg_computed_layer"]
                    with open(result_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([t, acc, layer, computed_layer, timee])
                        f.flush()
                    t = min(t + step, target)
                d += 1 
    elif cfg.get("exit_cl"):
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (valid_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)

        for select_last in ("True", "False"):
            os.environ["SELECT_LAST"] = select_last
            result_file = cfg.get("result_file")
            root, ext = os.path.splitext(result_file)
            if select_last == "True":
                result_file = f"{root}_last{ext}"
            else:
                result_file = f"{root}_max{ext}"
            ## Getting min layer ##
            t = 0
            os.environ["THRESHOLD"] = str(t)
            print(f"evaluating on threshold {t}")
            times = time.time()
            metric = solver.evaluate("test")
            timef = time.time()
            acc = metric[cfg.task.metric]
            timee = timef - times
            layer = metric["avg_layer"]
            computed_layer = metric["avg_computed_layer"]
            with open(result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["Threshold", cfg.task.metric, "Average Exit Layer", "Average Computed Layer", "Time"])
                writer.writerow([t, acc, layer, computed_layer, timee])

            ##Getting max layer ### 
            t = 1 
            os.environ["THRESHOLD"] = str(t)
            print(f"evaluating on threshold {t}")
            times = time.time()
            metric = solver.evaluate("test")
            timef = time.time()
            acc = metric[cfg.task.metric]
            timee = timef - times
            layer = metric["avg_layer"]
            computed_layer = metric["avg_computed_layer"]
            with open(result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([t, acc, layer, computed_layer, timee])
            max_layer = layer

            ##Iterating through ##
            t = 0
            layer = 0
            d = 1
            while layer < max_layer * .95: 
                target = 1 - 2*10**(-d)
                step = 2*10**(-d)
                i = 0
                while t < target: 
                    if i % 2 == 0: #doing half number of runs to save time
                        os.environ["THRESHOLD"] = str(t) 
                        times = time.time()
                        metric = solver.evaluate("test")
                        timef = time.time()
                        acc = metric[cfg.task.metric]
                        timee = timef - times
                        layer = metric["avg_layer"]
                        computed_layer = metric["avg_computed_layer"]
                        with open(result_file, 'a') as f:
                            writer = csv.writer(f)
                            writer.writerow([t, acc, layer, computed_layer, timee])
                            f.flush()
                    t = min(t + step, target)
                    i += 1
                d += 1 


    elif cfg.get("testing"):
        # if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
        #     cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
        #     train_set = core.Configurable.load_config_dict(cfg.dataset)
        #     cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
        #     valid_set = core.Configurable.load_config_dict(cfg.dataset)
        #     cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
        #     test_set = core.Configurable.load_config_dict(cfg.dataset)
        #     dataset = (valid_set, valid_set, test_set)
        # else:
        #     dataset = core.Configurable.load_config_dict(cfg.dataset)
        # solver, scheduler = util.build_downstream_solver(cfg, dataset)
        # os.environ["SELECT_LAST"] = "True"
        # result_file = cfg.get("result_file")
        # test_set = core.Configurable.load_config_dict(cfg.dataset)
        # solver.test_set = test_set


        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (train_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)
        os.environ["SELECT_LAST"] = "True"
        result_file = cfg.get("result_file")

        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cfg.task.metric, "Average Exit Layer", "Average Computed Layer"])
        os.environ["PERCENT"] = str(0)
        os.environ["THRESHOLD"] = str(0)
        metric = solver.evaluate("test")
        acc = metric[cfg.task.metric]
        layer = metric["avg_layer"]
        computed_layer = metric["avg_computed_layer"]
        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([acc, layer, computed_layer])
        os.environ["PERCENT"] = str(1)
        os.environ["THRESHOLD"] = str(1)
        metric = solver.evaluate("test")
        acc = metric[cfg.task.metric]
        layer = metric["avg_layer"]
        computed_layer = metric["avg_computed_layer"]
        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([acc, layer, computed_layer])

        os.environ["SELECT_LAST"] = "False"


        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([cfg.task.metric, "Average Exit Layer", "Average Computed Layer"])
        os.environ["PERCENT"] = str(0)
        os.environ["THRESHOLD"] = str(0)
        metric = solver.evaluate("test")
        acc = metric[cfg.task.metric]
        layer = metric["avg_layer"]
        computed_layer = metric["avg_computed_layer"]
        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([acc, layer, computed_layer])
        os.environ["PERCENT"] = str(1)
        os.environ["THRESHOLD"] = str(1)
        metric = solver.evaluate("test")
        acc = metric[cfg.task.metric]
        layer = metric["avg_layer"]
        computed_layer = metric["avg_computed_layer"]
        with open(result_file, 'a') as f:
            writer = csv.writer(f)
            writer.writerow([acc, layer, computed_layer])



        # with open(cfg.get("result_file"), 'w') as f:
        #     writer = csv.writer(f)
        #     writer.writerow(["Layer", cfg.task.metric, "Time"])
        #     for layer in range(0, int(cfg.get("num_layers"))):
        #         os.environ["LAYER"] = str(layer)
        #         times = time.time()
        #         results = solver.evaluate("test")   
        #         timef = time.time()   
        #         timee = timef - times
        #         metric = results[cfg.task.metric]  
        #         writer.writerow([layer, metric, timee])
        #         f.flush()


        # with open(result_file, 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([cfg.task.metric, "Average Exit Layer", "Average Computed Layer"])
        # os.environ["PERCENT"] = str(0)
        # os.environ["THRESHOLD"] = str(0)
        # metric = solver.evaluate("test")
        # acc = metric[cfg.task.metric]
        # layer = metric["avg_layer"]
        # computed_layer = metric["avg_computed_layer"]
        # with open(result_file, 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([acc, layer, computed_layer])
        # os.environ["PERCENT"] = str(1)
        # os.environ["THRESHOLD"] = str(1)
        # metric = solver.evaluate("test")
        # acc = metric[cfg.task.metric]
        # layer = metric["avg_layer"]
        # computed_layer = metric["avg_computed_layer"]
        # with open(result_file, 'a') as f:
        #     writer = csv.writer(f)
        #     writer.writerow([acc, layer, computed_layer])
    elif cfg.get("exit_node"):
        if cfg.dataset["class"] in ["EC", "GO", "MyFold"]:
            cfg.dataset.split = "training" if cfg.dataset["class"] == "MyFold" else "train"
            train_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "validation" if cfg.dataset["class"] == "MyFold" else "valid"
            valid_set = core.Configurable.load_config_dict(cfg.dataset)
            cfg.dataset.split = "test_fold" if cfg.dataset["class"] == "MyFold" else "test"
            test_set = core.Configurable.load_config_dict(cfg.dataset)
            dataset = (train_set, valid_set, test_set)
        else:
            dataset = core.Configurable.load_config_dict(cfg.dataset)
        solver, scheduler = util.build_downstream_solver(cfg, dataset)

        for select_last in ("True", "False"):
            os.environ["SELECT_LAST"] = select_last
            result_file = cfg.get("result_file")
            root, ext = os.path.splitext(result_file)
            if select_last == "True":
                result_file = f"{root}_last{ext}"
            else:
                result_file = f"{root}_max{ext}"
            with open(result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow(["Percent", "Threshold", cfg.task.metric, "Average Exit Layer", "Average Computed Layer", "Time"])

            os.environ["PERCENT"] = str(0)
            os.environ["THRESHOLD"] = str(0)
            times = time.time()
            metric = solver.evaluate("test")
            timef = time.time()
            timee = timef - times
            acc = metric[cfg.task.metric]
            layer = metric["avg_layer"]
            computed_layer = metric["avg_computed_layer"]
            with open(result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([0, 0, acc, layer, computed_layer, timee])


            os.environ["PERCENT"] = str(1)
            os.environ["THRESHOLD"] = str(1)
            times = time.time()
            metric = solver.evaluate("test")
            timef = time.time()
            timee = timef - times
            acc = metric[cfg.task.metric]
            layer = metric["avg_layer"]
            computed_layer = metric["avg_computed_layer"]
            with open(result_file, 'a') as f:
                writer = csv.writer(f)
                writer.writerow([1, 1, acc, layer, computed_layer, timee])

            for p in np.arange(.8, 1, .1):
                os.environ["PERCENT"] = str(p)
                avg_layer = 0
                t = 1 
                os.environ["THRESHOLD"] = str(t)
                print(f"evaluating on threshold {t} percent {p}")
                times = time.time()
                metric = solver.evaluate("test")
                timef = time.time()
                acc = metric[cfg.task.metric]
                timee = timef - times
                layer = metric["avg_layer"]
                computed_layer = metric["avg_computed_layer"]
                with open(result_file, 'a') as f:
                    writer = csv.writer(f)
                    writer.writerow([p, t, acc, layer, computed_layer, timee])
                max_layer = layer
                t = 0
                layer = 0

                t = 0
                while layer < 1:
                    t += .1
                    os.environ["THRESHOLD"] = str(t) 
                    times = time.time()
                    metric = solver.evaluate("test")
                    timef = time.time()
                    acc = metric[cfg.task.metric]
                    timee = timef - times
                    layer = metric["avg_layer"]
                    computed_layer = metric["avg_computed_layer"]
                    with open(result_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([p, t, acc, layer, computed_layer, timee])
                        f.flush()
                while layer < max_layer*.95:
                    t += .01
                    os.environ["THRESHOLD"] = str(t) 
                    times = time.time()
                    metric = solver.evaluate("test")
                    timef = time.time()
                    acc = metric[cfg.task.metric]
                    timee = timef - times
                    layer = metric["avg_layer"]
                    computed_layer = metric["avg_computed_layer"]
                    with open(result_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([p, t, acc, layer, computed_layer, timee])
                        f.flush()
                while t < 1:
                    t += .1
                    os.environ["THRESHOLD"] = str(t) 
                    times = time.time()
                    metric = solver.evaluate("test")
                    timef = time.time()
                    acc = metric[cfg.task.metric]
                    timee = timef - times
                    layer = metric["avg_layer"]
                    computed_layer = metric["avg_computed_layer"]
                    with open(result_file, 'a') as f:
                        writer = csv.writer(f)
                        writer.writerow([p, t, acc, layer, computed_layer, timee])
                        f.flush()

                # for t in np.concatenate([np.arange(0.0, 0.4, 0.2),np.arange(0.4, 1.0, 0.05)]):
                # #for t in range(0, .4, .2) and range(.4, 1, .05):
                #     os.environ["THRESHOLD"] = str(t) 
                #     times = time.time()
                #     metric = solver.evaluate("test")
                #     timef = time.time()
                #     acc = metric[cfg.task.metric]
                #     timee = timef - times
                #     layer = metric["avg_layer"]
                #     computed_layer = metric["avg_computed_layer"]
                #     with open(result_file, 'a') as f:
                #         writer = csv.writer(f)
                #         writer.writerow([p, t, acc, layer, computed_layer, timee])
                #         f.flush()

                # while layer < max_layer * .95: 
                #     target = 1 - 2*10**(-d)
                #     step = 2*10**(-d)
                #     while t < target: 
                #         os.environ["THRESHOLD"] = str(t) 
                #         times = time.time()
                #         metric = solver.evaluate("test")
                #         timef = time.time()
                #         acc = metric[cfg.task.metric]
                #         timee = timef - times
                #         layer = metric["avg_layer"]
                #         computed_layer = metric["avg_computed_layer"]
                #         with open(result_file, 'a') as f:
                #             writer = csv.writer(f)
                #             writer.writerow([p, t, acc, layer, computed_layer, timee])
                #             f.flush()
                #         t = min(t + step, target)
                #     d += 1 




import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import torch
from scipy.stats import pearsonr
from multiprocessing import Pool
from functools import partial, lru_cache
import scipy.linalg

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, default = None, help="Directory containing input .pt files.")
    parser.add_argument("--output_dir", type=str, required=False, default = None, help="Directory to save output files.")
    parser.add_argument("--run", type=str, default="", required=False)
    parser.add_argument("--consistency_file", type=str, default = "", required=False)
    parser.add_argument("--model_list", required=False, default=None, help="which set of models to run, see code")
    return parser.parse_args()


def get_low_rank_attention(a, b, c):
    n = b.shape[0]
    i_indices = np.arange(n)[:, None]  # shape (n, 1)
    j_indices = np.arange(n)[None, :]  # shape (1, n)
    relative_positions = i_indices - j_indices  # shape (n, n)
    L = a[relative_positions] + b[None, :] + c[:, None]
    return L

# attn[i, j] = attn(q_i, k_j)
# a: positional
# b: semantic
# c: nothing


def get_simulated_data(n):
    np.random.seed(0)
    a = np.random.randn(2*n-1)
    b = np.random.randn(n)
    c = np.random.randn(n)
    L = get_low_rank_attention(a, b, c)
    print("true a: ", a)
    print("true b: ", b)
    print("true c: ", c)
    print("L: ", L)
    return a, b, c, L

import numpy as np
import scipy.linalg
from functools import lru_cache

@lru_cache(maxsize=None)
def _prefactor_A(n: int):
    """
    Build A_reg (depends only on n), LU-factorize it once, and cache the factorization.
    """
    A = np.zeros((4 * n - 1, 4 * n - 1), dtype=np.float64)

    def index_a(i: int) -> int:
        return (2 * n + i - 1) if i < 0 else i

    def index_b(i: int) -> int:
        return 2 * n + i - 1

    def index_c(i: int) -> int:
        return 3 * n + i - 1

    # Fill A (same structure as your original code, but without y)
    for i in range(-n + 1, n):
        A[index_a(i), index_a(i)] = n - abs(i)
        if i >= 0:
            A[index_a(i), index_b(0): index_b(n - i)] = 1
            A[index_a(i), index_c(i): index_c(n)] = 1
        else:
            A[index_a(i), index_b(-i): index_b(n)] = 1
            A[index_a(i), index_c(0): index_c(n + i)] = 1

    for i in range(n):
        # b rows
        A[index_b(i), index_a(-i): index_a(-1)] = 1
        A[index_b(i), index_a(0): index_a(n - i)] = 1
        A[index_b(i), index_b(i)] = n
        A[index_b(i), index_c(0): index_c(n)] = 1

        # c rows
        A[index_c(i), index_a(i - n): index_a(-1)] = 1
        A[index_c(i), index_a(0): index_a(i)] = 1
        A[index_c(i), index_b(0): index_b(n)] = 1
        A[index_c(i), index_c(i)] = n

    A_reg = A + 1e-6 * np.eye(A.shape[0], dtype=A.dtype)
    lu, piv = scipy.linalg.lu_factor(A_reg)
    return lu, piv

def disentangle_matrix(L):
    """
    Faster drop-in replacement: reuses cached LU factorization for each n.
    L: (n, n) numpy array
    Returns: a, b, c
    """
    n = L.shape[0]
    y = np.zeros(4 * n - 1, dtype=np.float64)

    def index_a(i: int) -> int:
        return (2 * n + i - 1) if i < 0 else i

    def index_b(i: int) -> int:
        return 2 * n + i - 1

    def index_c(i: int) -> int:
        return 3 * n + i - 1

    # Fill y (same as your original code)
    for i in range(-n + 1, n):
        y[index_a(i)] = L.diagonal(-i).sum()

    for i in range(n):
        y[index_b(i)] = L[:, i].sum()
        y[index_c(i)] = L[i, :].sum()

    lu, piv = _prefactor_A(n)
    x = scipy.linalg.lu_solve((lu, piv), y)

    a = x[:index_a(-1) + 1]
    b = x[index_b(0): index_b(n)]
    c = x[index_c(0): index_c(n)]
    return a, b, c

def compute_components_and_consistency(all_logits):
    all_components = []
    correlations = []
    for layer_logits in all_logits:  # [n_heads, n, n]
        layer_components = []
        for head_logits in layer_logits:  # [n, n]
            a, b, c = disentangle_matrix(head_logits)
            layer_components.append([a, b, c])
            reconstructed = get_low_rank_attention(a, b, c)
            corr, _ = pearsonr(head_logits.ravel(), reconstructed.ravel())
            correlations.append(float(corr))
        all_components.append(np.array(layer_components, dtype=object))
    all_components = np.array(all_components, dtype=object)
    mean_corr = float(np.mean(correlations)) if correlations else float("nan")
    return all_components, correlations, mean_corr


def load_logits_as_lhnn(input_path, model):
    """
    Returns numpy array shaped [layers, heads, n, n] (float32/float64).
    """
    if model in ["Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_bert", "Rostlab/prot_xlnet", "Rostlab/prot_albert"]:
        x = torch.load(input_path, map_location="cpu")  # tensor
        x = x.permute(1, 0, 2, 3, 4)  # [1, layers, heads, n, n]
        x = x[0]  # [layers, heads, n, n]
        return x.numpy()

    if model == "t5-3b":
        x = torch.load(input_path, map_location="cpu")  # tuple(list) of layers
        x = torch.stack(x, dim=0).squeeze(1)            # [layers, heads, n, n]
        return x.numpy()

    if model in ["google-bert/bert-base-uncased", "albert-xxlarge-v2"]:
        x = torch.load(input_path, map_location="cpu")  # tuple of layer tensors
        x = torch.cat(x, dim=0)                         # [layers, heads, n, n]
        return x.numpy()

    if model == "xlnet/xlnet-large-cased":
        x = torch.load(input_path, map_location="cpu")  # tuple of layers
        x = torch.stack(x, dim=0).squeeze(1)            # [layers, heads, n, n]
        return x.numpy()

    if model == "esm":
        x = torch.load(input_path, map_location="cpu")
        # your current code uses all_logits[0]; keep that behavior:
        if isinstance(x, (tuple, list)):
            x = x[0]
        return x.numpy()

    raise ValueError(f"Unknown model: {model}")



def process_one_file(input_file, output_file, model, consistency_csv_path=None):
    # Skip if already processed
    if os.path.exists(output_file):
        return

    all_logits = load_logits_as_lhnn(input_file, model)  # [L, H, n, n]
    comps, corrs, mean_corr = compute_components_and_consistency(all_logits)

    with open(output_file, "wb") as f:
        pickle.dump(comps, f)

    # Optional: append consistency results
    if consistency_csv_path:
        import csv
        # Write one row per file; correlations list can be long, but it’s what you did before
        with open(consistency_csv_path, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([model, os.path.basename(input_file), mean_corr, corrs])


def process_folder(output_dir, input_dir, model, consistency_csv_path):
    os.makedirs(output_dir, exist_ok=True)
    i = 0
    for file_name in os.listdir(input_dir):
        if (i % 50 == 0):
            print(i)
        i+=1
        input_file = os.path.join(input_dir, file_name)
        output_file = os.path.join(output_dir, file_name.replace(".pt", "_processed.pkl"))
        process_one_file(input_file, output_file, model, consistency_csv_path=consistency_csv_path)
    return

def main(args):
    import random
    import csv
    results = {}
    model_list = ["google-bert/bert-base-uncased", "Rostlab/prot_bert", "albert-xxlarge-v2" , "Rostlab/prot_albert", "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50", "t5-3b", "Rostlab/prot_xlnet", "xlnet/xlnet-large-cased"]
    for model in model_list:
        print(f"processing model {model}")
        model_dirname = model.replace("/", "_")
        input_dir = os.path.join(args.input_dir, model_dirname)
        output_dir = os.path.join(args.output_dir, model_dirname)
        process_folder(output_dir, input_dir, model, args.consistency_file)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)

import numpy as np
import os
import pickle
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch
from scipy.stats import pearsonr

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True, help="Directory containing input .pt files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save output files.")
    parser.add_argument("--model", type=str, default="esm2", help = "The model used")
    parser.add_argument("--testing_consistency", default = False, required=False)
    return parser.parse_args()


def get_low_rank_attention(a, b, c):
    n = b.shape[0]
    L = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            L[i, j] = a[i-j] + b[j] + c[i]
    return L


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


def disentangle_matrix(L):
    n = L.shape[0]
    A = np.zeros((4*n-1, 4*n-1))
    y = np.zeros(4*n-1)

    def index_a(i):
        if i < 0:
            return 2 * n + i - 1
        else:
            return i

    def index_b(i):
        return 2 * n + i - 1

    def index_c(i):
        return 3 * n + i - 1

    for i in range(-n+1, n):
        # a related equations
        A[index_a(i), index_a(i)] = n - abs(i)
        if i >= 0:
            A[index_a(i), index_b(0): index_b(n-i)] = 1
            A[index_a(i), index_c(i): index_c(n)] = 1
        else:
            A[index_a(i), index_b(-i): index_b(n)] = 1
            A[index_a(i), index_c(0): index_c(n+i)] = 1
        y[index_a(i)] = L.diagonal(-i).sum()

    for i in range(n):
        # b related equations
        A[index_b(i), index_a(-i): index_a(-1)] = 1
        A[index_b(i), index_a(0): index_a(n-i)] = 1
        A[index_b(i), index_b(i)] = n
        A[index_b(i), index_c(0): index_c(n)] = 1
        y[index_b(i)] = L[:, i].sum()

        # c related equations
        A[index_c(i), index_a(i-n): index_a(-1)] = 1
        A[index_c(i), index_a(0): index_a(i)] = 1
        A[index_c(i), index_b(0): index_b(n)] = 1
        A[index_c(i), index_c(i)] = n
        y[index_c(i)] = L[i, :].sum()

    A_reg = A + 1e-6 * np.eye(A.shape[0])
    x = np.linalg.solve(A_reg, y)
    a = x[:index_a(-1)+1]
    b = x[index_b(0): index_b(n)]
    c = x[index_c(0): index_c(n)]
    return a, b, c



def process_tensor_file_esm2(input, output):
    #BERT: 24 layers, 16 attention heads
    # length 24 ([1, 16, 512, 512])

    import torch
    all_logits = torch.load(input)
    all_logits = all_logits.numpy()
    if len(all_logits.shape) == 5:
            all_logits = all_logits[0]

    all_components = []


   # print(all_logits.shape)
    for head_i, head_logits in enumerate(all_logits):
        a, b, c = disentangle_matrix(head_logits)
        all_components.append([a, b, c])  # Keep as list to handle different shapes
    all_components = np.array(all_components, dtype=object)
   # with open(output, 'wb') as f:
   #     pickle.dump(all_components, f)
    return all_components




def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    i = 0
    output_folder = args.output_dir
    os.makedirs(output_folder, exist_ok=True)
    for folder_name in os.listdir(args.input_dir):
        input_folder = os.path.join(args.input_dir, folder_name)

        # Ensure it's a directory (skip files)
        if not os.path.isdir(input_folder):
            continue

        # Create a corresponding output folder
        #output_folder = os.path.join(args.output_dir, folder_name)
        output_file = os.path.join(output_folder, f"{folder_name}_processed.pkl")
          # Create output folder if it doesn't exist

        print(f"Processing folder: {input_folder} -> Output file: {output_file}")
        all_components = []
        # Process each file in the current input folder
        i = 0
        for file_name in os.listdir(input_folder):
            if i % 20 == 0:
                print(i)
            i += 1
            if file_name.endswith(".pt"):  # Only process .pt files
                input_file = os.path.join(input_folder, file_name)
                components = process_tensor_file_esm2(input_file, output_file)
                all_components.append(components)
        all_components = np.array(all_components, dtype=object)
        with open(output_file, 'wb') as f:
            pickle.dump(all_components, f)
        

    print("Processing complete!")



if __name__ == '__main__':
    args = parse_args()
    main(args)
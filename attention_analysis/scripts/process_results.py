import os
import pickle
import torch
import matplotlib.pyplot as plt
import numpy as np
import json
from argparse import ArgumentParser
from collections import defaultdict
import matplotlib.colors as mcolors
from scipy.stats import chi2

def get_attention_variation_ratio(head, device='cpu'):
    """
    Compute the attention variation ratio for a given logit array.
    """
    std_p = torch.var(head[0]).to(device)
    std_s = torch.var(head[1]).to(device)
    ratio = (std_p / std_s).item()
    return ratio


def get_attention_variation_ratio_three_component(a, b, c, device='cpu'):
    var_s = torch.var(torch.tensor(b, dtype=torch.float32)).to(device)
    var_p = torch.var(torch.tensor(a, dtype=torch.float32)).to(device)
    ratio = (var_p / var_s).item()
    return ratio 

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--distribution_file", type=str, required=True, 
                        help="Where to store output JSON")
    parser.add_argument("--image_file", type=str, required=True, 
                        help="Where to store image file")
    parser.add_argument("--stats_output", type=str, required=True)
    parser.add_argument("--process_decomposed_files", default=False)
    return parser.parse_args()

def process_layer(layer_logits):
    """
    Process a single logit file and return the counts for positional, semantic, and mixed heads.
    """
    ratios = []

    for head_index, head in enumerate(layer_logits):
        #print(f'head shape {head.shape}')
        #print(head)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        ratio = get_attention_variation_ratio(torch.tensor(head), device)
        ratios.append(ratio)
    return ratios

def process_layer_three_component(layer_logits):
    ratios = []
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    for head_index, head in enumerate(layer_logits):
        a, b, c = head  
        ratio = get_attention_variation_ratio_three_component(a, b, c, device)
        ratios.append(ratio)
    return ratios
    

def process_protein(logit_file):
    """
    Process all layers (logit files) for a given protein folder.
    """
    layer_ratios = {}
    with open(logit_file, 'rb') as f:
        logits = pickle.load(f)
    #print(logits)
    #print(f"logits shape {logits.shape}")
    for layer_idx, layer_logit in enumerate(logits):
        layer_ratios[f"layer{layer_idx}"] = process_layer_three_component(layer_logit) #old way: process_layer
    return layer_ratios



def process_all_proteins(input):
    """
    Process all protein folders to generate the final dictionary.
    """
    i = 0
    protein_dict = {}
    for logit_file in os.listdir(input):
        i += 1
        logit_path = os.path.join(input, logit_file)
        protein_dict[logit_file] = process_protein(logit_path)
    return protein_dict

def flatten_ratios(data_dict):
    """
    Takes a dictionary like { "filename": { "layer0": [ratios...], "layer1": [ratios...] } }
    and flattens it into two lists:
      x_vals (layer indices repeated) and y_vals (the ratio values).
    """
    layer_ratios = defaultdict(list)
    for protein in data_dict:
        for layer_name, ratio_list in data_dict[protein].items():
            layer_ratios[layer_name].extend(ratio_list)

    all_layers = sorted(layer_ratios.keys(), key=lambda x: int(x.replace("layer", "")))
    x_vals, y_vals = [], []
    for layer_name in all_layers:
        layer_idx = int(layer_name.replace("layer", ""))
        for ratio in layer_ratios[layer_name]:
            x_vals.append(layer_idx)
            y_vals.append(ratio)
    
    if not x_vals or not y_vals:
        raise ValueError("Flattened data is empty, possibly due to incorrect input structure.")
    
    return x_vals, y_vals


def variance_confidence_interval(variance_array, n, confidence=0.95):
        """
        Given an array of sample variances (variance_array) computed with ddof=1
        and the sample size n along the axis we took the variance over,
        return (lower_bound, upper_bound) arrays of the same shape, giving
        the chi-square-based confidence interval for each variance.
        
        Assumes normality, i.e. s^2 * (n-1)/sigma^2 ~ chi2_{n-1}.
        """
        dof = n - 1  
        alpha = 1 - confidence  

        lower_q = 1 - alpha/2   
        upper_q = alpha/2       
        
        chi2_lower = chi2.ppf(lower_q, dof)  
        chi2_upper = chi2.ppf(upper_q, dof) 

        lower_bound = (dof * variance_array) / chi2_lower
        upper_bound = (dof * variance_array) / chi2_upper
        
        return (lower_bound, upper_bound)
    


def compute_ratio_statistics(data_dict, stats_file, name):
    proteins = list(data_dict.keys())
    layers = list(data_dict[proteins[0]].keys())
    num_proteins = len(proteins)
    num_layers = len(layers)
    num_heads = len(data_dict[proteins[0]][layers[0]])
    data_array = np.zeros((num_proteins, num_layers, num_heads), dtype=float)
    for i, p in enumerate(proteins):
        for j, l in enumerate(layers):
            head_list = data_dict[p].get(l, [])
            data_array[i,j,:] = head_list
    data_array = np.log(data_array)
    #by protein
    mean_data = np.abs(np.mean(data_array))
    variance_proteins = np.var(data_array, axis=0).mean()
    var_prot_lower, var_prot_upper = variance_confidence_interval(variance_proteins, n=num_proteins)
    cv_proteins = np.sqrt(variance_proteins) / mean_data
    cv_prot_lower = np.sqrt(var_prot_lower) / mean_data
    cv_prot_upper = np.sqrt(var_prot_upper) / mean_data
    #by layer 
    variance_layers = np.var(data_array, axis=1).mean()
    var_layers_lower, var_layers_upper = variance_confidence_interval(variance_layers, n=num_layers)
    cv_layers = np.sqrt(variance_layers) / mean_data
    cv_layer_lower = np.sqrt(var_layers_lower) / mean_data
    cv_layer_upper = np.sqrt(var_layers_upper) / mean_data
    #by head
    variance_heads = np.var(data_array, axis=2).mean()
    var_heads_lower, var_heads_upper = variance_confidence_interval(variance_heads, n=num_heads)
    cv_heads = np.sqrt(variance_heads) / mean_data
    cv_heads_lower = np.sqrt(var_heads_lower) / mean_data
    cv_heads_upper = np.sqrt(var_heads_upper) / mean_data
    with open(stats_file, "a") as file:
        print(f"Name {name} \n CV by input {cv_proteins} CI {cv_prot_lower} {cv_prot_upper}", file=file)
        print(f"CV by layers {cv_layers} CI {cv_layer_lower} {cv_layer_upper}", file=file)
        print(f"CV by heads {cv_heads} {cv_heads_lower} {cv_heads_upper}", file=file)
    #print(f"variance by proteins {variance_proteins} CI {var_prot_lower} {var_prot_upper}")
    #print(f"variance by layers {variance_layers} CI {var_layers_lower} {var_layers_upper}")
    #print(f"variance by heads {variance_heads} CI {var_heads_lower} {var_heads_upper}")
        
def visualize_ratios_heatmap(data_dict, image_file, title, global_min, global_max, y_axis):
    """
    Visualize a 2D histogram (heatmap) of attention ratios for each layer in data_dict.
    - data_dict has format { "layer0": [ratios...], "layer1": [ratios...], ... }
    - image_file is where the figure is saved
    - global_min and global_max define the shared y-scale for the heatmap
    """
    layer_ratios = defaultdict(list)
    for protein in data_dict:
        for layer_name, ratios_list in data_dict[protein].items():
            layer_ratios[layer_name].extend(ratios_list)

    # Convert layer_name (e.g. "layer0") to an integer index
    all_layers = sorted(layer_ratios.keys(), key=lambda x: int(x.replace("layer", "")))

    x_vals = []
    y_vals = []
    for layer_name in all_layers:
        # Extract the integer index from the layer_name
        layer_idx = int(layer_name.replace("layer", ""))
        # Append each ratio to y_vals; the x-value is the layer index
        for ratio in layer_ratios[layer_name]:
            x_vals.append(layer_idx)
            y_vals.append(ratio)
    log_bins = np.logspace(np.log10(global_min), np.log10(global_max), 50)


    # Create a 2D histogram; you can adjust bins as desired
    plt.figure(figsize=(6,5))
    plt.subplots_adjust(left=0.15)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#440154")
    cmap.set_under(color="#440154")
    plt.hist2d(x_vals, y_vals, bins=[len(all_layers), log_bins], cmap=cmap, norm=mcolors.LogNorm())
    plt.yscale('log')
    cbar = plt.colorbar(label='Attention Head Count')
    cbar.ax.tick_params(labelsize=14)
    if not y_axis:
        cbar.set_label('Attention Head Count', fontsize=16)
    else:
        cbar.set_label('')
    plt.xlabel('Layer', fontsize=16)
    if y_axis:
        plt.ylabel('Ratio', fontsize=16)
        plt.yticks(fontsize=14)
    else:
        plt.ylabel('')
        plt.yticks([])
        plt.tick_params(axis='y', left=False)

    plt.title(title, fontsize=20)
    plt.xticks(fontsize=14)
    
    plt.tight_layout()
    plt.savefig(image_file, dpi=500)
    plt.close()



def main():
    args = parse_args()

    pairs = [("albert-xxlarge-v2", "Rostlab_prot_albert"), ("google-bert_bert-base-uncased", "Rostlab_prot_bert"), ("t5-3b", "Rostlab_prot_t5_xl_bfd"), ("t5-3b", "Rostlab_prot_t5_xl_uniref50"), ("xlnet_xlnet-large-cased", "Rostlab_prot_xlnet")]
    title = {"albert-xxlarge-v2": "Albert", 
             "Rostlab_prot_albert": "ProtAlbert", 
             "google-bert_bert-base-uncased": "BERT",
             "Rostlab_prot_bert": "ProtBERT", 
             "t5-3b": "T5", 
             "Rostlab_prot_t5_xl_bfd": "ProtT5 (BFD)",
             "Rostlab_prot_t5_xl_uniref50": "ProtT5 (UniRef50)",
             "xlnet_xlnet-large-cased": "XLNet",
             "Rostlab_prot_xlnet": "ProtXLNet"
             }
        # We want the same y-axis scale for both protein and NLP plots
        # Flatten each to find min and max

    for nlp, protein in pairs: 
        print(f"Processing files {nlp} and {protein}")
        protein_folder = os.path.join(args.input_folder, protein)
        nlp_folder = os.path.join(args.input_folder, nlp)

        nlp_distribution_file = os.path.join(args.distribution_file, nlp)
        protein_distribution_file = os.path.join(args.distribution_file, protein)

        image_file_protein = os.path.join(args.image_file, protein)
        image_file_nlp = os.path.join(args.image_file, nlp)

        if args.process_decomposed_files:
            protein_data = process_all_proteins(protein_folder)
            nlp_data = process_all_proteins(nlp_folder)
            with open(protein_distribution_file, "w") as f:
                json.dump(protein_data, f, indent=4)
            with open(nlp_distribution_file, "w") as f:
                json.dump(nlp_data, f, indent=4)
        else:
            with open(protein_distribution_file, "r") as f:
                protein_data = json.load(f)
            with open(nlp_distribution_file, "r") as f:
                nlp_data = json.load(f)
        _, protein_ratios = flatten_ratios(protein_data)
        _, nlp_ratios = flatten_ratios(nlp_data)
    
        overall_min = min(min(protein_ratios), min(nlp_ratios))
        overall_max = max(max(protein_ratios), max(nlp_ratios))


        # Visualize protein
        visualize_ratios_heatmap(
            protein_data, 
            image_file=image_file_protein,
            title=title[protein],
            global_min=overall_min,
            global_max=overall_max,
            y_axis=False
        )

        # Visualize NLP
        visualize_ratios_heatmap(
            nlp_data,
            image_file=image_file_nlp,
            title=title[nlp],
            global_min=overall_min,
            global_max=overall_max,
            y_axis=True
        )

        compute_ratio_statistics(protein_data, args.stats_output, protein)
        compute_ratio_statistics(nlp_data, args.stats_output, nlp)


if __name__ == '__main__':
    main()

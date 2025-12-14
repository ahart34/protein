import os
os.environ['HF_HOME'] = '/scratch/anna19/cache'
os.environ['TORCH_HOME'] = '/scratch/anna19/cache'
import torch
from transformers import T5Tokenizer, T5EncoderModel, AutoModel, AutoTokenizer, AlbertTokenizer, XLNetTokenizer
import esm
import json
from argparse import ArgumentParser
def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--attn_dir", type=str, required=True, help="Directory to put attention files")
    parser.add_argument("--data_file", type=str, required=True, help="File of protein sequences")
    return parser.parse_args()

def all_models(sequence_file, model_name, attn_dir):
    # Load ProtTrans model and tokenizer (replace with desired model)
    #model_name = args.model #"Rostlab/prot_t5_xl_uniref50"  # Change this to the desired ProtTrans model
    if (model_name == "Rostlab/prot_t5_xl_uniref50") or (model_name == "Rostlab/prot_t5_xl_bfd"):
        tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        model = T5EncoderModel.from_pretrained(model_name, output_attentions=True)
    elif (model_name == "Rostlab/prot_albert"):
        tokenizer = AlbertTokenizer.from_pretrained(model_name, cache_dir="/scratch/anna19/cache/", use_fast=False, do_lower_case=False, do_basic_tokens=False)
        model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir="/scratch/anna19/cache/")
    elif (model_name == "Rostlab/prot_xlnet"):
        tokenizer = XLNetTokenizer.from_pretrained("Rostlab/prot_xlnet", cache_dir="/scratch/anna19/cache/")
        model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir="/scratch/anna19/cache/")
    elif (model_name == "esm"):
        model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        batch_converter = alphabet.get_batch_converter()
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/anna19/cache/")
        model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir="/scratch/anna19/cache/")
    model.eval()  # Disable dropout for deterministic results
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Load the input data
    with open(sequence_file, 'r') as f:
        data = json.load(f)
        data_list = [(entry[0], entry[1]) for entry in data]

    # Process each sequence in the dataset
    i=0
    for entry in data_list:
        if model_name == "esm":
            batch_labels, batch_strs, batch_tokens = batch_converter([entry])
            batch_tokens = batch_tokens[:, :512].to(device)
            with torch.no_grad():
                results = model(batch_tokens, repr_layers=[33], need_head_weights=True)
                attn = results["attentions"]
            if i == 0:
                print(f"attention shape {attn.shape}")
            torch.save(attn, os.path.join(attn_dir, f"Attention_results_{i}.pt"))
        else:
            sequence_id, sequence = entry
            print(len(sequence))
            sequence_spaced = " ".join(list(sequence))
            # Tokenize the sequence
            inputs = tokenizer(sequence_spaced, return_tensors="pt", padding=True, truncation=True, max_length=512) #512 added for max length for XLnet
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                # Forward pass through the model to get attentions
                outputs = model(**inputs)
                attentions = outputs.attentions  # Tuple of attention tensors (one per layer)

            attentions_tensor = torch.stack(attentions)  

            torch.save(attentions_tensor, os.path.join(attn_dir, f"Attention_results_{i}.pt"))
            if i == 0: 
                print(attentions_tensor.shape)
        i+=1

def main(args):
    for model in ["Rostlab/prot_bert", "Rostlab/prot_albert", "Rostlab/prot_t5_xl_bfd", "Rostlab/prot_t5_xl_uniref50", "Rostlab/prot_xlnet", "esm"]:
        print(f"processing model {model}")
        model_dirname = model.replace("/", "_")
        attn_dir = os.path.join(args.attn_dir, model_dirname)
        os.makedirs(attn_dir, exist_ok=True)
        all_models(args.data_file, model, attn_dir)
    return

if __name__ == '__main__':
    args = parse_args()
    main(args)

#Prot_t5_xl_bfd attention dimensions: torch.Size([24, 1, 32, 490, 490])
#ProtAlbertALBERT: torch.Size([12, 1, 64, 130, 130])
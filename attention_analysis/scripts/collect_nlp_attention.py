from datasets import load_dataset, load_from_disk
import os
os.environ['HF_HOME'] = '/scratch/anna19/cache'
os.environ['TORCH_HOME'] = '/scratch/anna19/cache'
from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
import torch
import json
import os
from argparse import ArgumentParser
import itertools
from datasets.download import DownloadConfig


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--attn_dir", type=str, required=True, help="Directory to put attention files")
    parser.add_argument("--data_file", type=str, required=True, help="File to save metadata")
    return parser.parse_args()

def all_models(dataset, model_name, data_file, attn_dir):
    print("Loading model")
    # Load T5 model with attention outputs enabled
    print(model_name)
    if (model_name == "t5-3b"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/anna19/cache/")
        model = T5EncoderModel.from_pretrained(model_name, output_attentions=True, cache_dir="/scratch/anna19/cache/")
    elif (model_name == "albert-xxlarge_v2"):
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/anna19/cache/")
        model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir="/scratch/anna19/cache/")
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/scratch/anna19/cache/")
        model = AutoModel.from_pretrained(model_name, output_attentions=True, cache_dir="/scratch/anna19/cache/")
    print("Finished loading model")

    
    model.eval()  # Set model to evaluation mode
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Define output directories
    base_dir = attn_dir

    input_metadata = []
    # Process each entry to get attention weights
    for i, entry in enumerate(dataset):
        if i % 50 == 0:
            print(f"Processing input {i}")
        inputs = tokenizer(entry["text"], return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_length = len(inputs['input_ids'][0])
        input_metadata.append({
            "index": i,
            "text": entry["text"],
            "tokenized_length": input_length
        })

        # Run the model and collect attentions
        with torch.no_grad():
            outputs = model(**inputs)
        attentions = outputs.attentions  # Attention weights for all layers

        #How T5 attention is stored: tuple - 24, then attentions[0] = torch.size([1,32,n,n])

        # Save attention weights for each example
        torch.save(
            attentions,
            os.path.join(base_dir, f"Attention_results_{i}.pt")
        )

    print(f"Processing complete. Results saved to {base_dir}")
    with open(data_file, 'w') as f:
        json.dump(input_metadata, f, indent=4)

def main(args):
    dataset_full = load_dataset("DKYoon/SlimPajama-6B", split="train", cache_dir="/scratch/anna19/cache2")
    print("finished loading dataset")
    dataset_shuffled = dataset_full.shuffle(seed=10)
    print("finished shuffling")
    dataset = list(itertools.islice(dataset_shuffled, 1000))
    for model in ["google-bert/bert-base-uncased", "albert-xxlarge-v2", "t5-3b", "xlnet/xlnet-large-cased"]:
        print(f"processing model {model}")
        model_dirname = model.replace("/", "_")
        attn_dir = os.path.join(args.attn_dir, model_dirname)
        os.makedirs(attn_dir, exist_ok=True)
        data_file = os.path.join(args.data_file, f"{model_dirname}_data.json")
        all_models(dataset, model, data_file, attn_dir)
    return 


if __name__ == '__main__':
    args = parse_args()
    main(args)

import torch
from torch import nn
from torchdrug.core import Registry as R
from transformers import AutoTokenizer, AutoModel, AlbertTokenizer

@R.register("models.CustomProtBert")
class CustomProtBert(nn.Module):
    def __init__(self, output_dim=1024):
        super().__init__()
        print("Initializing custom ProtBert")
        self.tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False, cache_dir="/scratch/anna19/cache2/")
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert", cache_dir="/scratch/anna19/cache2/")
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.output_dim = output_dim
        self.num_layers = self.model.config.num_hidden_layers 

    def forward(self, sequences, repr_layers=None, per_residue=False):
        # Prepare inputs via tokenizer
        # Add special tokens ([CLS], [SEP]) automatically
        spaced_sequences = [" ".join(list(seq)) for seq in sequences]
        device = next(self.model.parameters()).device
        encoded_input = self.tokenizer(
            list(spaced_sequences),
            return_tensors="pt",
            padding=True,
            truncation=True,        # optional if you want to ensure max length
            add_special_tokens=True,
            max_length=552 #was 550 for EC, GO, CL
        ).to(device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=encoded_input["input_ids"],
                attention_mask=encoded_input["attention_mask"],
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states  

        if repr_layers is None:
            # Use the final hidden layer from huggingface (last index)
            last_layer = hidden_states[-1]  # shape: (batch_size, seq_len, hidden_dim)
            last_layer = last_layer[:, 1:-1, :]
            if per_residue:
                output = last_layer
            else:
                output = last_layer.mean(dim=1)  # average over seq_len
        else:
            output = {}
            for r in repr_layers:
                if r < 0 or r >= len(hidden_states):
                    raise ValueError(
                        f"Requested layer {r} is out of valid range 0..{len(hidden_states)-1} "
                        f"for this ProtBert model."
                    )
                out = hidden_states[r]
                out = out[:, 1:-1, :]
                if per_residue:
                    output[r] = out
                else:
                    output[r] = out.mean(dim=1)

        return output

    @classmethod
    def load_config_dict(cls, config_dict):
        return cls()

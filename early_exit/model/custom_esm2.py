import torch
import esm
from torch import nn
from torchdrug.core import Registry as R
import pickle

@R.register("models.CustomESM2")
class CustomESM2(nn.Module):
    def __init__(self, output_dim=1280): #t12_150: output_dim = 640; t_33_650: output_dim = 1280
        super(CustomESM2, self).__init__()
        print("initializing custom ESM2")
        # Load ESM2 model and alphabet
        self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()
        #self.model, self.alphabet = esm.pretrained.esm2_t30_150M_UR50D()
        self.model.eval()  # Disable dropout
        self.batch_converter = self.alphabet.get_batch_converter()
        for param in self.model.parameters():
            param.requires_grad = False
        # Set output dimension for downstream tasks
        self.output_dim = output_dim
        self.num_layers = self.model.num_layers

    def forward(self, sequences, repr_layers=None, obtain_attention=False, per_residue=False):
        data = [(f"protein_{i}", seq) for i, seq in enumerate(sequences)]
        _, _, batch_tokens = self.batch_converter(data)

        device = next(self.model.parameters()).device
        batch_tokens = batch_tokens.to(device)

        with torch.no_grad():
            if obtain_attention:
                if repr_layers == None:
                    results = self.model(batch_tokens, return_contacts=False, repr_layers=[self.model.num_layers], need_head_weights=True)
                    output = results
                else:
                    results = self.model(batch_tokens, return_contacts=False, repr_layers=repr_layers, need_head_weights= True)
                    output = results
            else:
                if per_residue:
                    if repr_layers == None:
                        results = self.model(batch_tokens, return_contacts=False, repr_layers=[self.model.num_layers])
                        hidden_states = results["representations"][self.model.num_layers]
                        output = hidden_states #Shape: (batch_size, hidden_dim)
                    else:
                        results = self.model(batch_tokens, return_contacts=False, repr_layers=repr_layers)
                        hidden_states = {}
                        for r in repr_layers:
                            hidden_states[r] = results["representations"][r]
                        output = hidden_states
                else:
                    if repr_layers == None:
                        results = self.model(batch_tokens, return_contacts=False, repr_layers=[self.model.num_layers])
                        hidden_states = results["representations"][self.model.num_layers]
                        output = hidden_states.mean(dim=1) #Shape: (batch_size, hidden_dim)
                    else:
                        results = self.model(batch_tokens, return_contacts=False, repr_layers=repr_layers)
                        hidden_states = {}
                        for r in repr_layers:
                            hidden_states[r] = results["representations"][r].mean(dim=1)
                        output = hidden_states
        
        return output


    @classmethod
    def load_config_dict(cls, config_dict):
        return cls()
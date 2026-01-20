# Data Download: 
## Attention analysis:
- Run collect_protein_sequences.py to collect 1,000 proteins from UniProt
- Run collect_nlp_attention.py: 1,000 NLP samples will be automatically downloaded from SlimPajama-6B

## Early Exit: 
- To download GO and EC datasets, Follow "Download Datasets and Model Weights" from Zhang et al. Github: https://github.com/DeepGraphLearning/esm-s/tree/main
- CL and SSP datasets will download automatically when running algorithms 


# Running Code: 
## Attention Analysis: 
- Run collect_protein_sequences.py to collect proteins from UniProt
- Run collect_nlp_attention.py and collect_protein_attention.py to collect attention heads of NLP and PLM models for samples
  - python attention_analysis/scripts/collect_protein_attention.py --attn_dir raw_attention --data_file uniprot_sequences.json
  - python attention_analysis/scripts/collect_nlp_attention.py --attn_dir raw_attention --data_file metadata_file_output

- Run compute_attention_components.py to decompose the attention heads
  - python attention_analysis/scripts/compute_attention_components.py --input_dir /shared/nas2/anna19/protein/attention_analysis/results/raw_attention --output_dir /shared/nas2/anna19/protein/attention_analysis/results/decomposed_attention
 
- Processing results:
  - python attention_analysis/scripts/process_results.py --input_folder attention_analysis/results/decomposed_attention
    - Set flags and output directories for the type and location of results you wish to output
   
## Early Exit: 
- .yaml files for configurations are stored in /config folder
- To run .yaml file:
  - For training: python -m torch.distributed.launch --nproc_per_node=4  ./script/run.py -c config/model/yaml_file --datadir data_dir --level bp --modeldir ./model
    - --level flag is used only for GO. --datadir flag is used only for GO and EC
  - For running early exit or evaluation:
    - CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.launch --nproc_per_node=1  ./script/run.py -c config/model/yaml_file --datadir data_dir --level bp --modeldir ./model
  - For running confidence analysis:
    - First run _confidence .yaml files
    - Then use evaluate_confidence.py and plot_confidence.py 

# Environment Setup: 
## Attention Analysis: 
conda env create -f attention_analysis/attention_analysis.yml
conda activate attention_analysis

## Early Exit: 
conda env create -f early_exit/early_exit.yml
conda activate early_exit

# References: 
This code utilizes the TorchDrug Framework: https://torchdrug.ai <br/>
Code for building MLP and handling GO and EC datasets utilizes aspects of code in https://github.com/DeepGraphLearning/esm-s/tree/main (Zhang et al. "Structure-Informed Protein Language Model") and code for run.py and handling CL and SSP datasets utilizes aspects of code in https://github.com/DeepGraphLearning/PEER_Benchmark/blob/main (Xu et al. "PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding") <br/>
Datasets GO, EC, CL, and SSP are obtained as stated in our paper and will be downloaded by following instructions above. 



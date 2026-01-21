# Data Download: 
## Attention analysis:
- Run collect_protein_sequences.py to collect 1,000 proteins from UniProt
- Run collect_nlp_attention.py: 1,000 NLP samples will be automatically downloaded from SlimPajama-6B

## Early Exit: 
- To download GO and EC datasets, Follow "Download Datasets and Model Weights" from Zhang et al. Github: https://github.com/DeepGraphLearning/esm-s/tree/main
- CL and SSP datasets will download automatically when running algorithms

## Data References: 
- UniProt: The UniProt Consortium, A. Bateman, M.-J. Martin,
S. Orchard, et al. UniProt: the universal protein
knowledgebase in 2023. Nucleic Acids Research, 51:D523–
D531, 2023.
- SlimPajama-6B: Dongkeyun Yoon. SlimPajama-6b. https://huggingface.co/
datasets/DKYoon/SlimPajama-6B, 2023.
- GO and EC:
  - Data split found in: Z. Zhang, J. Lu, V. Chenthamarakshan, et al. Structure-
informed protein language model. GEM Workshop, ICLR,
2024.
  - Original Data Source: V. Gligorijevi´c, P. D. Renfrew, T. Kosciolek, et al. Structure-
based protein function prediction using graph convolutional
networks. Nature Communications, 12(1):3168, 2021. ISSN
2041-1723.
- CL and SSP:
  - Data split found in: M. Xu, Z. Zhang, J. Lu, et al. PEER: A comprehensive and
multi-task benchmark for protein sequence understanding.
In Advances in Neural Information Processing Systems
35: Annual Conference on Neural Information Processing
Systems 2022, NeurIPS 2022, New Orleans, LA, USA,
November 28 - December 9, 2022, 2022.
  - Original Data Souce:
    - CL: J. J. Almagro Armenteros, C. K. Sønderby, S. K. Sønderby,
et al. DeepLoc: prediction of protein subcellular localization using deep learning. Bioinformatics, 33(21):3387–3395,
2017. ISSN 1367-4803, 1367-4811.
    - SSP: M. S. Klausen, M. C. Jespersen, H. Nielsen, et al. NetSurfP-
2.0: Improved prediction of protein structural features by
integrated deep learning. Proteins: Structure, Function,
and Bioinformatics, 87(6):520–527, 2019. ISSN 0887-3585,
1097-0134. J. A. Cuff and G. J. Barton. Evaluation and improvement of
multiple sequence methods for protein secondary structure
prediction. Proteins: Structure, Function, and Genetics,
34(4):508–519, 1999. ISSN 0887-3585, 1097-0134


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
Code for building MLP and handling GO and EC datasets utilizes aspects of code in https://github.com/DeepGraphLearning/esm-s/tree/main (Zhang et al. "Structure-Informed Protein Language Model") and code for run.py and handling CL and SSP datasets utilizes aspects of code in https://github.com/DeepGraphLearning/PEER_Benchmark/blob/main (Xu et al. "PEER: A Comprehensive and Multi-Task Benchmark for Protein Sequence Understanding") <br/>
This code utilizes TorchDrug https://torchdrug.ai <br/>
Datasets GO, EC, CL, and SSP are obtained as stated in our paper and will be downloaded by following instructions above. 





import os
import gzip
import random
import requests
from Bio import SeqIO
import json 

PROT_URL = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
FASTA_GZ_FILE = "/scratch/anna19/uniprot_sprot.fasta.gz"
FASTA_FILE = "/scratch/anna19/uniprot_sprot.fasta"
SAMPLED_FASTA = "/shared/nas2/anna19/protein/attention_analysis/results/swissprot_sampled_1000.fasta"
NUM_SAMPLES = 1000


def download_fasta():
    if os.path.exists(FASTA_GZ_FILE):
        print(f"File already exists: {FASTA_GZ_FILE}")
        return

    print(f"Downloading FASTA from {PROT_URL} ...")
    response = requests.get(PROT_URL, stream=True)
    response.raise_for_status()

    with open(FASTA_GZ_FILE, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

    print(f"Downloaded: {FASTA_GZ_FILE}")

def decompress_fasta():
    if os.path.exists(FASTA_FILE):
        print(f"Decompressed FASTA already exists: {FASTA_FILE}")
        return

    print(f"🗜️ Decompressing {FASTA_GZ_FILE} ...")
    with gzip.open(FASTA_GZ_FILE, "rt") as gz_file:
        with open(FASTA_FILE, "w") as out_file:
            for line in gz_file:
                out_file.write(line)

    print(f"Decompressed to: {FASTA_FILE}")

def sample_sequences():
    print(f"Reading FASTA entries from {FASTA_FILE} ...")
    records = list(SeqIO.parse(FASTA_FILE, "fasta"))
    print(f"Total sequences available {len(records)}")
    sampled = random.sample(records, NUM_SAMPLES)
    SeqIO.write(sampled, SAMPLED_FASTA, "fasta")
    print(f"Saved {NUM_SAMPLES} random sequences to {SAMPLED_FASTA}")
    json_output_path = SAMPLED_FASTA.replace(".fasta", ".json")
    json_data = [[record.id.split("|")[-1], str(record.seq)] for record in sampled]

    with open(json_output_path, "w") as jf:
        json.dump(json_data, jf, indent=4)

    print(f"Also saved {len(json_data)} JSON data to {json_output_path}")

if __name__ == "__main__":
    download_fasta()
    decompress_fasta()
    sample_sequences()



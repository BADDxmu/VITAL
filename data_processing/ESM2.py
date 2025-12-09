#!/usr/bin/env python3
import os
import sys
import subprocess


def esm():
    CURRENT_DIR = os.path.dirname(__file__)
    path_seq = os.path.join(CURRENT_DIR, 'tmp/fasta/sequence/')
    path_seg = os.path.join(CURRENT_DIR, 'tmp/fasta/fragment/')
    path_fasta = os.path.join(CURRENT_DIR, 'tmp/fasta/')

    def concatenate_files(file_list, output_file):
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for file_path in file_list:
                with open(file_path, 'r', encoding='utf-8') as infile:
                    outfile.write(infile.read())
                    # outfile.write('\n')  

    seq_list = os.listdir(path_seq)
    seg_list = os.listdir(path_seg)

    seq_list = [f"{path_seq}{file}" for file in seq_list]
    seg_list = [f"{path_seg}{file}" for file in seg_list]

    concatenate_files(seq_list, f"{path_fasta}sequence.fasta")
    concatenate_files(seg_list, f"{path_fasta}fragment.fasta")


    #--------------------------
    # python ESM2.py
    #--------------------------



    checkpoint_path = os.path.join(CURRENT_DIR, 'ESM-2/checkpoints/esm2_t30_150M_UR50D.pt')
    input_fasta_seq = f"{path_fasta}sequence.fasta"
    input_fasta_seg = f"{path_fasta}fragment.fasta"
    output_dir_seq = os.path.join(CURRENT_DIR, 'tmp/feature/esm2_sequence/')
    output_dir_seg = os.path.join(CURRENT_DIR, 'tmp/feature/esm2_fragment/')
    repr_layers = [0, 29, 30]
    include_mean = "mean"

    os.makedirs(output_dir_seq, exist_ok=True)
    os.makedirs(output_dir_seg, exist_ok=True)


    ### esm2:sequence ###
    # command
    command = [
        "python", os.path.join(CURRENT_DIR, 'ESM-2/scripts/extract.py'),
        "--model_location", checkpoint_path,
        "--fasta_file", input_fasta_seq,
        "--output_dir", output_dir_seq,
        "--repr_layers", *map(str, repr_layers),
        "--include", include_mean
    ]

    # 
    print("Running command: " + " ".join(command))

    # 
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        sys.exit(1)



    ### esm2:fragment ###
    # command
    command = [
        "python", os.path.join(CURRENT_DIR, 'ESM-2/scripts/extract.py'),
        "--model_location", checkpoint_path,
        "--fasta_file", input_fasta_seg,
        "--output_dir", output_dir_seg,
        "--repr_layers", *map(str, repr_layers),
        "--include", include_mean
    ]

    # 
    print("Running command: " + " ".join(command))

    # 
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error during execution: {e}")
        sys.exit(1)


if __name__ == "__main__":
    esm()
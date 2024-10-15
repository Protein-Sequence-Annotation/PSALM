#!/bin/bash

#SBATCH -J test
#SBATCH -p eddy
#SBATCH -c 16
#SBATCH -t 0-2:00
#SBATCH -o logs/val.out
#SBATCH -e logs/val.err
#SBATCH --mem 180000 #180GB

module load gcc #/13.2.0-fasrc01
module load Mambaforge

mamba deactivate
mamba activate protllm

# python val_creator.py

# # File paths
# KEYS_FILE="datasets/val_keys.txt"
# ORIGINAL_FASTA="datasets/Train_1b_merged.fasta"
# OUTPUT_FASTA="datasets/Val_1b_final.fasta"

# # Make sure the output file is empty initially
# > "$OUTPUT_FASTA"

# # Create an associative array to store the keys
# declare -A keys

# # Read keys from the file into the array
# while IFS= read -r key; do
#     keys["$key"]=1
# done < "$KEYS_FILE"

# # Variable to track if current entry should be included
# include=0

# # Read the FASTA file
# while IFS= read -r line; do
#     if [[ $line == \>* ]]; then # Header line
#         header="${line#>}"  # Remove the '>' character
#         if [[ -n "${keys[$header]}" ]]; then
#             include=1
#             echo "$line" >> "$OUTPUT_FASTA"
#         else
#             include=0
#         fi
#     else
#         if [[ $include -eq 1 ]]; then
#             echo "$line" >> "$OUTPUT_FASTA"
#         fi
#     fi
# done < "$ORIGINAL_FASTA"

# echo "Sampled FASTA entries saved to '$OUTPUT_FASTA'"

# Define file paths
KEYS_FILE="datasets/val_keys.txt"
INPUT_FASTA="datasets/Train_1b_merged_copy.fasta"
OUTPUT_FASTA="datasets/Train_1b_final.fasta"

# Create an associative array to store the keys
declare -A keys_to_remove

# Read keys from the file into the associative array
while IFS= read -r key; do
    keys_to_remove["$key"]=1
done < "$KEYS_FILE"

# Variable to track if current entry should be included
include=1

# Make sure the output file is empty initially
> "$OUTPUT_FASTA"

# Read and filter the FASTA file
while IFS= read -r line; do
    if [[ $line == \>* ]]; then # Header line
        header="${line#>}"  # Remove the '>' character
        if [[ -n "${keys_to_remove[$header]}" ]]; then
            include=0
        else
            include=1
            echo "$line" >> "$OUTPUT_FASTA"
        fi
    else # Sequence line
        if [[ $include -eq 1 ]]; then
            echo "$line" >> "$OUTPUT_FASTA"
        fi
    fi
done < "$INPUT_FASTA"

echo "Filtered FASTA file saved to '$OUTPUT_FASTA'"
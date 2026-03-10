

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                                                                              │
│                                                                              │
│                 ██████╗ ███████╗ █████╗ ██╗     ███╗   ███╗                  │
│                 ██╔══██╗██╔════╝██╔══██╗██║     ████╗ ████║                  │
│                 ██████╔╝███████╗███████║██║     ██╔████╔██║                  │
│                 ██╔═══╝ ╚════██║██╔══██║██║     ██║╚██╔╝██║                  │
│                 ██║     ███████║██║  ██║███████╗██║ ╚═╝ ██║                  │
│                 ╚═╝     ╚══════╝╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝                  │
│              Protein Sequence Annotation using a Language Model              │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```



Persistent session mode (load model once, scan many times):
```
psalm -d auto
# inside shell:
#   scan -f path/to/seqs.fa
#   scan --sort -f path/to/seqs.fa -c 4 --to-tsv hits.tsv
#   scan -s "MSTNPKPQR..."
#   quit
```

Quick usage:
```
psalm-scan -f path/to/your_sequence.fasta
```

CLI behavior notes:
- Default model: `ProteinSequenceAnnotation/PSALM-2`
- Default device: `auto` (`cuda` -> `mps` -> `cpu`)
- FASTA scans use fast batched scanning by default
- `--serial` restores the legacy serial FASTA behavior
- `--sort` remains opt-in
- `-c/--cpu-workers` is the number of fast-mode CPU decode helper processes
  - default behavior is equivalent to `-c 0`
  - if the interactive shell already has warmed workers, later default fast scans reuse that pool
- `--max-batch-size` controls the fast-mode embedding batch budget in tokens/amino acids
- `--max-queue-size` controls the fast-mode decode queue in sequences
  - default: `128`
- `-q/--quiet` suppresses scan result output only; startup/status still prints
- `--to-tsv` and `--to-txt` work for single or multi-sequence FASTA
- `-v/--verbose` enables detailed alignment and model tables
  - verbose FASTA scans use the serial path
  - without `-v`, PSALM prints the compact HITS report
- `-T` keeps domains with `Score >= threshold` (default: `0.5`)
- `-E` keeps domains with `E-value <= threshold` (default: `0.1`)
- `-Z` sets dataset size for E-value scaling
  - if omitted for `-s`: `Z=1`
  - if omitted for `-f`: `Z=#sequences in FASTA`
- `--to-tsv` is the supported machine-readable output format


Common shell usage:
```
psalm
scan --sort -f path/to/seqs.fa --to-tsv hits.tsv
```

Useful output modes:
```
# compact terminal report + TSV
scan -f path/to/seqs.fa --to-tsv hits.tsv

# with TSV only
scan -q --sort -f path/to/seqs.fa --to-tsv hits.tsv

# verbose per-domain output
scan -v -f path/to/seqs.fa
```

For the full option set, run `psalm --help`, `psalm-scan --help`, or `scan --help`.

## Installation

Create a fresh Python 3.10 environment, install PyTorch for your hardware, then install PSALM.
```
conda create -n psalm python=3.10 -y
conda activate psalm
python -m pip install --upgrade pip

# 1) Install PyTorch for your hardware
# Apple Silicon (MPS):
python -m pip install torch

# CPU-only (Linux/Windows):
# python -m pip install torch

# NVIDIA CUDA 12.1:
# python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
#   torch

# 2) Install PSALM
python -m pip install protein-sequence-annotation==2.1.9
```

If you are unsure which PyTorch command matches your GPU/driver, use the official selector:
https://pytorch.org/get-started/locally/

Intel Mac (x86_64) tested path:
```
conda create -n psalm python=3.10 -y
conda activate psalm

conda install -y -c conda-forge "llvmlite=0.44.*" "numba=0.61.*"
conda install -y -c conda-forge "pytorch=2.5" torchvision torchaudio

python -m pip install protein-sequence-annotation==2.1.9
```

Optional: run without activating conda manually:
```
conda run -n psalm psalm-scan -f path/to/seqs.fa
```

## Python API

```python
from psalm.psalm_model import PSALM

psalm = PSALM(model_name="ProteinSequenceAnnotation/PSALM-2")

# Scan FASTA
results = psalm.scan(fasta="path/to/your_sequence.fasta")
print(results)

# Scan sequence string
results = psalm.scan(sequence="MSTNPKPQR...AA")
```

Output options:
- `to_tsv="results.tsv"` writes:
  `Sequence,E-value,Score,Pfam,Start,Stop,Model,Len Frac,Status`
- `to_txt="results.txt"` saves console-style output
- For multi-sequence FASTA, TSV rows are combined with the query id in the `Sequence` column

## Scripts overview
The core workflow is:
1) `scripts/data/augment_fasta.py` → slice sequences and generate augmented FASTA + domain dict
2) `scripts/data/data_processing.py` → tokenize, label, batch, and shard datasets
3) `scripts/train/train_psalm.py` → train/evaluate the PSALM model on shards

### `scripts/data/augment_fasta.py`
Splits long sequences into domain-preserving slices and optionally emits shuffled
and negative variants. Produces a new FASTA and a new domain dict with aligned
IDs.

**Key inputs**
- `--fasta`, `--domain-dict`
- `--output-fasta`, `--output-dict`

**Common flags**
- `--max-length`: slice length threshold
- `--negative-prob`: target fraction of negatives (approximate)
- `--include-domain-slices`, `--shuffle-only`, `--no-shuffle`, `--domain-slices-only`
- `--large-data` with `--p-shuffled`, `--domain-counts-tsv`, `--domain-slice-frac`
- `--seed`, `--verbose`

### `scripts/data/data_processing.py`
Tokenizes sequences, generates per-token labels from the domain dict and label
mapping, batches by token budget, and saves shards.

**Config handling**
- This script is CLI-only; it does not read `config.yaml`.

**Required args**
- `--fasta`, `--domain-dict`, `--output-dir`, `--ignore-label`
- `--model-name`, `--max-length`, `--max-tokens-per-batch`
- `--label-mapping-dict`

**Optional args**
- `--chunk-size`, `--tmp-dir`, `--shard-size`, `--seed`, `--keep-tmp`

**Notes**
- ID normalization uses the FASTA header segment between `>` and the first space.
- `--ignore-label` must match the training `--ignore-label`.

### `scripts/train/train_psalm.py`
Trains or evaluates PSALM on preprocessed shard datasets.

**Config handling**
- Training always uses a YAML config.
- If `--config` is provided without a value, the script looks for
  `psalm/config.yaml`.
- If `--config` is not provided, the script still looks for
  `psalm/config.yaml`.

**Required args**
- `--val-dir`, `--ignore-label`
- `--train-dir` if `training.total_steps > 0` in config

**Optional args**
- `--label-mapping-dict` to override config `model.label_mapping_path`

**Checkpoint loading**
- Supports `model.safetensors` or `pytorch_model.bin` within a checkpoint
  directory, or a direct path to a `.safetensors`/`.bin` file.

**Logging**
- `report_to=["wandb"]` is enabled by default.

### `scripts/train/train_cbm.py`
Trains the CatBoost scoring model used by `scan()` (saved as `score.cbm`).

**Required args**
- `--pos`, `--neg`: Pickle or JSON files containing a list of 7-tuples:
  `(pfam, start, stop, bit_score, len_ratio, bias, status)`
  (or `scan()` output dicts containing 8-tuples with `cbm_score`).

**Example**
```
python scripts/train/train_cbm.py \
  --pos path/to/positives.pkl \
  --neg path/to/negatives.pkl \
  --outdir cbm_outputs \
  --model-out score.cbm
```

## Config format
The scripts expect a YAML config with these sections:

**`model`**
- `model_name`
- `max_batch_size`
- `output_size`
- `freeze_esm`
- `use_fa`
- `pretrained_checkpoint_path`
- `label_mapping_path`

**`training`**
- `gradient_accumulation_steps`, `learning_rate`, `optimizer`, `gradient_clipping`
- `lr_scheduler`, `eval_strategy`, `eval_steps`, `total_steps`, `warmup_steps`
- `logging_steps`, `save_steps`, `output_dir`
- `mixed_precision`, `dataloader_num_workers`, `dataloader_prefetch_factor`,
  `dataloader_pin_memory`, `seed`

**`data`**
- `chunk_size`, `default_tmp_dir`, `default_shard_size`

`psalm/config.yaml` is provided as a template with `null` values. Populate it
before use, or pass all required values via CLI without `--config`.

## Training CLI examples
```
python scripts/data/augment_fasta.py \
  --fasta input.fa \
  --domain-dict domains.pkl \
  --output-fasta augmented.fa \
  --output-dict augmented.pkl
```

```
python scripts/data/data_processing.py \
  --fasta augmented.fa \
  --domain-dict augmented.pkl \
  --label-mapping-dict labels.pkl \
  --output-dir data/shards \
  --model-name ProteinSequenceAnnotation/esm2_t33_650M_PFS90_leaky \
  --max-length 4096 \
  --max-tokens-per-batch 8196 \
  --ignore-label -100
```

```
python scripts/train/train_psalm.py \
  --config psalm/config.yaml \
  --train-dir data/shards/train \
  --val-dir data/shards/val \
  --ignore-label -100
```

## Dependencies
- `PyYAML` is required for config loading.
- `faesm` is required only if `use_fa: true` in config.
- Core inference runtime uses `torch`, `transformers`, `biopython`, `pandas`, `numba`, and `catboost`.

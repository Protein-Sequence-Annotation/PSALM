## **Important: PSALM is being heavily revamped**
This repository is under active refactor. For the previous stable codebase, see
https://github.com/Protein-Sequence-Annotation/PSALM-copy and release 1.13 or
previous.



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
                               PSALM CLI (v2.1.2)
```



Persistent session mode (load model once, scan many times):
```
psalm -d auto
# inside shell:
#   scan -f path/to/seqs.fa
#   scan -s "MSTNPKPQR..."
#   quit
```

Quick usage:
```
psalm-scan -f path/to/your_sequence.fasta
```

CLI behavior notes:
- Default model: `ProteinSequenceAnnotation/PSALM-2-InterPro30-104.0`
- Default device: `auto` (`cuda` -> `mps` -> `cpu`)
- `--quiet` suppresses scan result output only; startup/status still prints
- `--to-tsv` and `--to-txt` work for single or multi-sequence FASTA
- `-v/--verbose` enables detailed alignment and model tables
- `-E` keeps domains with `E-value <= threshold` (default: `0.01`)
- `-Z` sets dataset size for E-value scaling
  - if omitted for `-s`: `Z=1`
  - if omitted for `-f`: `Z=#sequences in FASTA`
- E-value values are computed from a packaged empirical negative-score interpolation curve
- HITS table is sorted by increasing `E-value`
- HITS and per-family tables show `Len Frac`
- `--to-tsv` is the supported machine-readable output format
- Model lookup order:
  1) explicit local path
  2) `models/<model_name>` under repository
  3) Hugging Face cache
  4) Hugging Face download (first run)

## Installation

### Conda environments (local development)
No CUDA:
```
conda env create -f psalm2.yml
conda activate psalm2
```

CUDA (NVIDIA GPU):
```
conda env create -f psalm2-cuda.yml
conda activate psalm2-cuda
```

### Install from latest TestPyPI build
```
python -m pip install --index-url https://test.pypi.org/simple/ \
  --extra-index-url https://pypi.org/simple \
  protein-sequence-annotation==2.1.2
```

Optional: run without activating conda manually:
```
conda run -n psalm2 psalm-scan -f path/to/seqs.fa
```

## Python API

```python
from psalm.psalm_model import PSALM

psalm = PSALM(model_name="ProteinSequenceAnnotation/PSALM-2-InterPro30-104.0")

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
  `src/psalm/config.yaml`.
- If `--config` is not provided, the script still looks for
  `src/psalm/config.yaml`.

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

`src/psalm/config.yaml` is provided as a template with `null` values. Populate it
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
  --config src/psalm/config.yaml \
  --train-dir data/shards/train \
  --val-dir data/shards/val \
  --ignore-label -100
```

## Dependencies
- `PyYAML` is required for config loading.
- `faesm` is required only if `use_fa: true` in config.
- Core inference runtime uses `torch`, `transformers`, `biopython`, `pandas`, `numba`, and `catboost`.

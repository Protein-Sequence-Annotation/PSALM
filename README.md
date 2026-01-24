## **Important: PSALM is being heavily revamped**
This repository is under active refactor. For the previous stable codebase, see
https://github.com/Protein-Sequence-Annotation/PSALM-copy and release 1.13 or
previous.

## Installation
Coming soon!


## Example usage
```python
from psalm.psalm_model import PSALM

# Load the model from HuggingFace
psalm = PSALM(model_name="ProteinSequenceAnnotation/PSALM-2-InterPro30-104.0")

# Scan from FASTA
results = psalm.scan(fasta="path/to/your_sequence.fasta")
print(results)
# {seq_id: [(pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status), ...]}

# Or scan from a raw sequence string
seq = "MSTNPKPQR...AA"
results = psalm.scan(sequence=seq)
```

Notes:git st
- For full usage (local bundles, HF download, verbose output), see `tutorial.ipynb`.
- `Score` in the output table is the CBM score (sorting/filtering).
- `Bit score` is the PF-LLD score (raw bit score from decoding).
- Optional knobs: `use_fa`, `beam_size`, `refine_extended`, `score_thresh`, `verbose`.

## Scripts overview
The core workflow is:
1) `augment_fasta.py` → slice sequences and generate augmented FASTA + domain dict
2) `data_processing.py` → tokenize, label, batch, and shard datasets
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

## Example usage
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
- Core runtime uses `torch`, `transformers`, and `datasets`.

#!/usr/bin/env python3
import glob
import os
import argparse
import torch.distributed as dist
import pickle
import sys
from pathlib import Path


def _add_src_to_path():
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))


def _resolve_config_path(config_arg):
    repo_root = Path(__file__).resolve().parents[2]
    default_config = repo_root / "psalm" / "config.yaml"
    if config_arg is None or config_arg == "__default__":
        if not default_config.exists():
            raise FileNotFoundError(
                "No --config provided and default config not found at "
                f"{default_config}. Provide --config explicitly."
            )
        return str(default_config)
    if not Path(config_arg).exists():
        raise FileNotFoundError(f"Config file not found: {config_arg}")
    return config_arg


def main():
    _add_src_to_path()
    from psalm.config import get_training_config, get_model_config, set_config
    from psalm.psalm_model import PSALM
    from psalm.train.psalm_trainer import PSALMTrainer
    from transformers import DefaultDataCollator, TrainingArguments

    parser = argparse.ArgumentParser(
        description="Train PSALM + finetune ESM-2 on preprocessed protein domain annotation datasets."
    )
    parser.add_argument(
        "--config",
        nargs="?",
        const="__default__",
        default=None,
        help="Path to a YAML config (use without a value to load psalm/config.yaml)",
    )
    parser.add_argument(
        "--train-dir",
        type=str,
        required=False,
        help="Path to the processed training dataset (arrow shards). Not required if total_steps=0.",
    )
    parser.add_argument(
        "--val-dir",
        type=str,
        required=True,
        help="Path to the processed validation dataset (a single HF dataset folder).",
    )
    parser.add_argument(
        "--label-mapping-dict",
        type=str,
        default=None,
        help="Optional path to label mapping pickle (overrides config if provided).",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        required=True,
        help="Label value used for ignored tokens (e.g., -100).",
    )
    args = parser.parse_args()

    config_path = _resolve_config_path(args.config)
    set_config(config_path)

    print(sys.executable)
    print(sys.path)

    is_distributed = dist.is_initialized()
    rank = dist.get_rank() if is_distributed else 0

    training_config = get_training_config()

    train_shard_paths = None
    if training_config["total_steps"] > 0:
        if args.train_dir is None:
            raise ValueError("--train-dir is required when total_steps > 0")
        train_dir = args.train_dir
        train_shard_paths = sorted(glob.glob(os.path.join(train_dir, "shard-*")))
        if len(train_shard_paths) == 0:
            raise ValueError(f"No shards found in {train_dir}!")
        if rank == 0:
            print(f"Found {len(train_shard_paths)} shards for training in {train_dir}.")
    else:
        if rank == 0:
            print("total_steps is 0, skipping training data loading...")

    val_dir = args.val_dir
    val_shard_paths = sorted(glob.glob(os.path.join(val_dir, "shard-*")))
    if len(val_shard_paths) == 0:
        raise ValueError(f"No shards found in {val_dir}!")
    if rank == 0:
        print(f"Found {len(val_shard_paths)} shards for validation in {val_dir}.")

    if rank == 0:
        print("Initializing model and tokenizer...")

    cfg = get_model_config()
    if hasattr(cfg, "pretrained_checkpoint_path") and cfg.pretrained_checkpoint_path:
        if rank == 0:
            print(f"Loading model from checkpoint: {cfg.pretrained_checkpoint_path}")
        model = PSALM.from_checkpoint(cfg.pretrained_checkpoint_path, ignore_label=args.ignore_label)
    else:
        model = PSALM(ignore_label=args.ignore_label)

    if cfg.use_fa and training_config["total_steps"] == 0:
        model = model.half()
        if rank == 0:
            print("Converted model to half precision for FlashAttention evaluation")

    label_mapping_path = args.label_mapping_dict
    if not label_mapping_path:
        label_mapping_path = getattr(cfg, "label_mapping_path", None)

    if label_mapping_path:
        if rank == 0:
            print(f"Loading label mapping from: {label_mapping_path}")
        with open(label_mapping_path, "rb") as f:
            label_mapping = pickle.load(f)
        model.label_mapping = label_mapping
        if rank == 0:
            print("Label mapping loaded successfully for metrics")

    data_collator = DefaultDataCollator()

    training_args = TrainingArguments(
        output_dir=training_config["output_dir"],
        overwrite_output_dir=True,
        eval_strategy=training_config["eval_strategy"],
        eval_steps=training_config["eval_steps"],
        save_steps=training_config["save_steps"],
        max_grad_norm=training_config["gradient_clipping"],
        lr_scheduler_type=training_config["lr_scheduler"],
        warmup_steps=training_config["warmup_steps"],
        max_steps=training_config["total_steps"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        logging_dir="./logs",
        logging_steps=training_config["logging_steps"],
        report_to=["wandb"],
        fp16=(training_config["mixed_precision"] == "fp16"),
        dataloader_num_workers=training_config["dataloader_num_workers"],
        dataloader_pin_memory=training_config["dataloader_pin_memory"],
        dataloader_prefetch_factor=training_config["dataloader_prefetch_factor"],
        seed=training_config["seed"],
        save_safetensors=False,
        ddp_find_unused_parameters=True,
    )

    trainer = PSALMTrainer(
        train_dataset=train_shard_paths,
        eval_dataset=val_shard_paths,
        data_collator=data_collator,
        gradient_clipping=training_config["gradient_clipping"],
        ignore_label=args.ignore_label,
        model=model,
        args=training_args,
    )

    if rank == 0:
        print("Starting training...")

    if training_config["total_steps"] == 0:
        if rank == 0:
            print("total_steps is 0, running evaluation only...")
        eval_results = trainer.evaluate()
        if rank == 0:
            print("Evaluation results:")
            for key, value in eval_results.items():
                print(f"  {key}: {value}")
    else:
        trainer.train()


if __name__ == "__main__":
    main()

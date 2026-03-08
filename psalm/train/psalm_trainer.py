import math
import random
import torch
from torch.utils.data import get_worker_info, DataLoader, IterableDataset, DistributedSampler, Dataset
from collections import defaultdict
from transformers import Trainer
from datasets import load_from_disk
from tqdm import tqdm
import gc


def fine_to_coarse(labels_fine, none_val=0, pad_val=-100):
    """
    Converts fine labels in the range [0, N] to coarse labels in the range [0, N/3],
    while properly handling padding tokens.
    """
    coarse = torch.full_like(labels_fine, pad_val)
    valid_mask = labels_fine != pad_val
    none_mask = labels_fine == none_val
    coarse[valid_mask & none_mask] = none_val
    coarse[valid_mask & (~none_mask)] = ((labels_fine[valid_mask & (~none_mask)] - 1) // 3) + 1
    return coarse


class PreloadedBatchDataset(Dataset):
    """
    A regular Dataset that preloads all shards, groups into batches, and supports
    the same functionality as ShardBatchIterable but with better performance.
    """

    def __init__(
        self,
        shard_paths,
        group_by_batch_fn,
        seed,
        rank,
        world_size,
        args,
        sort_batches_by_length=True,
        epoch=0,
        shuffle=True,
    ):
        super().__init__()
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.sort_batches_by_length = sort_batches_by_length
        self.epoch = epoch
        self.shuffle = shuffle

        if self.rank == 0:
            print(f"[Rank {self.rank}] Preloading {len(shard_paths)} shards...")

        all_examples = []
        for shard_idx, shard_path in enumerate(shard_paths):
            if self.rank == 0:
                print(f"[Rank {self.rank}] Loading shard {shard_idx + 1}/{len(shard_paths)}: {shard_path}")

            shard_data = load_from_disk(shard_path)
            all_examples.extend(list(shard_data))

        if self.rank == 0:
            print(f"[Rank {self.rank}] Grouping {len(all_examples)} examples into batches...")

        self.batches = group_by_batch_fn(all_examples)

        if self.sort_batches_by_length:
            if self.rank == 0:
                print(f"[Rank {self.rank}] Sorting {len(self.batches)} batches by length...")
            self.batches.sort(key=lambda batch: len(batch[-1]["input_ids"]))

        groups = [
            self.batches[i: i + self.world_size]
            for i in range(0, len(self.batches), self.world_size)
        ]
        rng = random.Random(self.seed + int(self.epoch))
        if self.shuffle:
            rng.shuffle(groups)
        self.batches = [batch for group in groups for batch in group]

        if self.rank == 0:
            print(f"[Rank {self.rank}] Dataset ready with {len(self.batches)} batches")

    def __len__(self):
        return len(self.batches)

    def __getitem__(self, idx):
        return self.batches[idx]


class ShardBatchIterable(IterableDataset):
    """
    An IterableDataset that processes one shard at a time, sorts batches by length,
    and distributes batches across GPUs using distributed sampling behavior.
    """

    def __init__(
        self,
        shard_paths,
        group_by_batch_fn,
        epoch,
        seed,
        rank,
        world_size,
        args,
        sort_batches_by_length=True,
        shuffle=True,
    ):
        super().__init__()
        self.shard_paths = shard_paths
        self.group_by_batch_fn = group_by_batch_fn
        self.epoch = epoch
        self.real_epoch = 0
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.args = args
        self.sort_batches_by_length = sort_batches_by_length
        self.shuffle = shuffle

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id = 0
            num_workers = 1
        else:
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        rng = random.Random(self.seed + int(self.epoch))
        shards = self.shard_paths[:]
        if self.shuffle:
            rng.shuffle(shards)

        if self.rank == 0:
            print(f"[Rank {self.rank}] Starting epoch {self.epoch}")

        for shard_idx, shard_path in enumerate(shards):
            if self.rank == 0:
                print(f"[Rank {self.rank}] Loading shard {shard_idx + 1}/{len(shards)}: {shard_path}")

            shard_data = load_from_disk(shard_path)
            examples = list(shard_data)
            batches = self.group_by_batch_fn(examples)

            if self.sort_batches_by_length:
                batches.sort(key=lambda batch: len(batch[-1]["input_ids"]))

            groups = [
                batches[i: i + self.world_size]
                for i in range(0, len(batches), self.world_size)
            ]
            if self.shuffle:
                rng.shuffle(groups)
            batches = [batch for group in groups for batch in group]

            total_batches = len(batches)
            batches_per_worker = total_batches // num_workers
            worker_batches = batches[
                worker_id * batches_per_worker: (worker_id + 1) * batches_per_worker
            ]
            sampler = DistributedSampler(
                worker_batches,
                num_replicas=self.world_size,
                rank=self.rank,
                shuffle=False,
            )
            sampler.set_epoch(self.real_epoch)
            for batch_idx in sampler:
                yield worker_batches[batch_idx]

            del worker_batches, batches, examples, shard_data, groups, sampler
            gc.collect()

        self.real_epoch += 1


class PSALMTrainer(Trainer):
    """
    Custom Trainer for sharded batches with category-specific evaluation metrics.
    """

    def __init__(self, train_dataset, eval_dataset, data_collator, gradient_clipping, ignore_label, *args, **kwargs):
        super().__init__(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            *args,
            **kwargs,
        )
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.gradient_clipping = gradient_clipping
        self.ignore_label = ignore_label
        self.true_global_step = 0

    def group_by_batch(self, dataset):
        grouped_data = defaultdict(list)
        for example in dataset:
            grouped_data[example["batch_id"]].append(example)
        return list(grouped_data.values())

    @staticmethod
    def create_collate_fn(base_collator, ignore_label, keys_to_remove=None):
        if keys_to_remove is None:
            keys_to_remove = []

        def custom_collate_fn(batch_list):
            lumps = batch_list[0]
            seq_ids = [ex.get("id", f"unknown_{i}") for i, ex in enumerate(lumps)]
            seq_lengths = [len(ex["input_ids"]) for ex in lumps]

            for ex in lumps:
                for key in keys_to_remove:
                    if key in ex:
                        ex.pop(key, None)

            batch = base_collator(lumps)
            batch["seq_ids"] = seq_ids
            batch["seq_lengths"] = seq_lengths

            flat_labels = batch["labels"].view(-1)
            unique_labels = torch.unique(flat_labels)
            if unique_labels.numel() == 1 and unique_labels.item() == ignore_label:
                raise ValueError(
                    "All labels in the collated batch are ignore_label. "
                    "Check your domain_dict and label_mapping inputs."
                )
            return batch

        return custom_collate_fn

    def get_train_dataloader(self):
        if not isinstance(self.train_dataset, list):
            return super().get_train_dataloader()

        epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        collate_fn = self.create_collate_fn(
            base_collator=self.data_collator,
            ignore_label=self.ignore_label,
            keys_to_remove=["batch_id", "sequence_length"],
        )

        if len(self.train_dataset) == 1:
            dataset = PreloadedBatchDataset(
                shard_paths=self.train_dataset,
                group_by_batch_fn=self.group_by_batch,
                seed=self.args.seed,
                rank=self.args.local_rank,
                world_size=self.args.world_size,
                args=self.args,
                sort_batches_by_length=True,
                epoch=epoch,
                shuffle=True,
            )

            sampler = DistributedSampler(
                dataset,
                num_replicas=self.args.world_size,
                rank=self.args.local_rank,
                shuffle=False,
            )
            sampler.set_epoch(epoch)

            return DataLoader(
                dataset,
                batch_size=1,
                sampler=sampler,
                collate_fn=collate_fn,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                prefetch_factor=self.args.dataloader_prefetch_factor,
            )

        shard_iterable = ShardBatchIterable(
            shard_paths=self.train_dataset,
            group_by_batch_fn=self.group_by_batch,
            epoch=epoch,
            seed=self.args.seed,
            rank=self.args.local_rank,
            world_size=self.args.world_size,
            args=self.args,
            sort_batches_by_length=True,
            shuffle=True,
        )

        return DataLoader(
            shard_iterable,
            batch_size=1,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
            prefetch_factor=self.args.dataloader_prefetch_factor,
        )

    def get_eval_dataloader(self, eval_dataset=None):
        eval_dataset = eval_dataset or self.eval_dataset
        collate_fn = self.create_collate_fn(
            base_collator=self.data_collator,
            ignore_label=self.ignore_label,
            keys_to_remove=["batch_id", "sequence_length"],
        )

        if isinstance(eval_dataset, list):
            all_examples = []
            for shard_path in eval_dataset:
                ds = load_from_disk(shard_path)
                all_examples.extend(list(ds))
        else:
            all_examples = list(eval_dataset)

        batches = self.group_by_batch(all_examples)
        batches.sort(key=lambda batch: len(batch[-1]["input_ids"]))

        sampler = DistributedSampler(
            batches,
            num_replicas=self.args.world_size,
            rank=self.args.local_rank,
            shuffle=False,
            drop_last=False,
        )

        return DataLoader(
            batches,
            batch_size=1,
            sampler=sampler,
            collate_fn=collate_fn,
            num_workers=self.args.dataloader_num_workers,
        )

    def training_step(self, model, inputs, num_items_in_batch=None):
        loss = super().training_step(model, inputs, num_items_in_batch)
        loss = loss / self.args.gradient_accumulation_steps
        return loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        filtered_inputs = {
            k: v for k, v in inputs.items()
            if k not in ["seq_ids", "seq_lengths"]
        }
        return super().compute_loss(model, filtered_inputs, return_outputs, num_items_in_batch)

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        eval_dataset = eval_dataset or self.eval_dataset
        if eval_dataset is None:
            return {}

        loader = self.get_eval_dataloader(eval_dataset)
        model = self.model.to(self.args.device).eval()
        ignore_label = self.ignore_label

        categories = ["originals", "shuffled", "domain_slices", "negatives"]
        cat_metrics = {
            cat: {"loss": 0.0, "n_loss": 0, "tpr1": 0, "tpr5": 0, "n_tpr": 0, "fpr1": 0, "n_fpr": 0}
            for cat in categories
        }

        pnone_medians = {cat: [] for cat in categories}
        pnone_all = []
        agg_metrics = {"loss": 0.0, "n_loss": 0, "tpr1": 0, "tpr5": 0, "n_tpr": 0, "fpr1": 0, "n_fpr": 0}
        total_coarse_label_sets = 0
        included_coarse_labels = 0

        debug_counts = {cat: 0 for cat in categories}
        debug_sample_ids = {cat: [] for cat in categories}

        with torch.no_grad():
            if self.args.local_rank in [-1, 0]:
                progress_bar = tqdm(loader, desc="Evaluating", leave=False)
            else:
                progress_bar = loader

            for batch in progress_bar:
                seq_ids = batch.get("seq_ids", [])
                seq_lengths = batch.get("seq_lengths", [])

                batch_tensors = {
                    k: v.to(self.args.device) for k, v in batch.items()
                    if isinstance(v, torch.Tensor) and k not in ["seq_ids", "seq_lengths"]
                }
                out = model(**batch_tensors)
                batch_loss = out["loss"] if isinstance(out, dict) else out
                logits = out["logits"]
                labels = batch_tensors["labels"]

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignore_label, reduction="none")
                B, L, C = logits.shape
                per_example_losses = loss_fct(logits.view(-1, C), labels.view(-1))
                per_example_losses = per_example_losses.view(B, L)

                example_losses = []
                for i in range(B):
                    valid_mask = labels[i] != ignore_label
                    if valid_mask.sum() > 0:
                        example_losses.append(per_example_losses[i][valid_mask].mean().item())
                    else:
                        example_losses.append(0.0)

                for i, seq_id in enumerate(seq_ids):
                    seq_len = seq_lengths[i] if i < len(seq_lengths) else logits.shape[1]
                    clean_id = seq_id[1:] if seq_id.startswith(">") else seq_id

                    if clean_id.startswith("negative_"):
                        cat = "negatives"
                    elif clean_id.startswith("shuffled_"):
                        cat = "shuffled"
                    elif "/" in clean_id:
                        cat = "domain_slices"
                    else:
                        cat = "originals"

                    debug_counts[cat] += 1
                    if len(debug_sample_ids[cat]) < 3:
                        debug_sample_ids[cat].append(clean_id)

                    ex_logits = logits[i]
                    ex_labels = labels[i]
                    ex_loss = example_losses[i] if i < len(example_losses) else 0.0
                    agg_metrics["loss"] += ex_loss
                    agg_metrics["n_loss"] += 1
                    cat_metrics[cat]["loss"] += ex_loss
                    cat_metrics[cat]["n_loss"] += 1

                    flat_logits = ex_logits.view(-1, ex_logits.size(-1))
                    flat_labels = ex_labels.view(-1)
                    valid_mask = flat_labels != ignore_label
                    flat_logits = flat_logits[valid_mask]
                    flat_labels = flat_labels[valid_mask]

                    preds_top1 = flat_logits.argmax(dim=-1)
                    preds_top5 = flat_logits.topk(5, dim=-1).indices

                    bg_label = 0
                    is_domain = flat_labels != bg_label
                    is_background = flat_labels == bg_label

                    n_true = is_domain.sum().item()
                    if n_true:
                        tpr1 = (preds_top1[is_domain] == flat_labels[is_domain]).sum().item()
                        tpr5 = (preds_top5[is_domain] == flat_labels[is_domain].unsqueeze(1)).any(dim=1).sum().item()
                        agg_metrics["tpr1"] += tpr1
                        agg_metrics["tpr5"] += tpr5
                        agg_metrics["n_tpr"] += n_true
                        cat_metrics[cat]["tpr1"] += tpr1
                        cat_metrics[cat]["tpr5"] += tpr5
                        cat_metrics[cat]["n_tpr"] += n_true

                    try:
                        probs = torch.softmax(ex_logits, dim=-1)
                        p_none_tokens = probs[:, 0]
                        bg_mask = (ex_labels == 0)
                        if bg_mask.any():
                            med_val = p_none_tokens[bg_mask].median().item()
                            pnone_medians[cat].append(med_val)
                            pnone_all.append(med_val)
                    except RuntimeError:
                        l0 = ex_logits[:, 0]
                        max_logit, _ = ex_logits.max(dim=-1, keepdim=True)
                        exp0 = (l0 - max_logit.squeeze(-1)).exp()
                        denom = (ex_logits - max_logit).exp().sum(dim=-1)
                        p_none_tokens = exp0 / denom.clamp_min(1e-12)
                        bg_mask = (ex_labels == 0)
                        if bg_mask.any():
                            med_val = p_none_tokens[bg_mask].median().item()
                            pnone_medians[cat].append(med_val)
                            pnone_all.append(med_val)

                    if cat == "negatives":
                        n_tokens = flat_labels.numel()
                        fpr1 = (preds_top1 != bg_label).sum().item()
                        agg_metrics["fpr1"] += fpr1
                        agg_metrics["n_fpr"] += n_tokens
                        cat_metrics[cat]["fpr1"] += fpr1
                        cat_metrics[cat]["n_fpr"] += n_tokens
                    elif cat == "shuffled":
                        n_bg = is_background.sum().item()
                        if n_bg > 0:
                            fpr1 = (preds_top1[is_background] != bg_label).sum().item()
                            agg_metrics["fpr1"] += fpr1
                            agg_metrics["n_fpr"] += n_bg
                            cat_metrics[cat]["fpr1"] += fpr1
                            cat_metrics[cat]["n_fpr"] += n_bg

                    coarse_labels = fine_to_coarse(flat_labels, pad_val=ignore_label)
                    valid_coarse = coarse_labels != ignore_label
                    coarse_labels = coarse_labels[valid_coarse]

                    if coarse_labels.numel() > 0:
                        unique_coarse = torch.unique(coarse_labels)
                        total_coarse_label_sets += 1

                        coarse_logits = flat_logits[valid_coarse]
                        coarse_preds_top1 = coarse_logits.argmax(dim=-1)
                        coarse_preds_fine = fine_to_coarse(coarse_preds_top1, pad_val=ignore_label)

                        if torch.any(torch.isin(coarse_preds_fine, unique_coarse)):
                            included_coarse_labels += 1

        results = {}
        for cat in categories:
            n_loss = cat_metrics[cat]["n_loss"]
            n_tpr = cat_metrics[cat]["n_tpr"]
            n_fpr = cat_metrics[cat]["n_fpr"]
            results[f"{cat}_loss"] = cat_metrics[cat]["loss"] / n_loss if n_loss else float("nan")
            if cat in ["originals", "shuffled", "domain_slices"]:
                results[f"{cat}_tpr1"] = cat_metrics[cat]["tpr1"] / n_tpr if n_tpr else float("nan")
                results[f"{cat}_tpr5"] = cat_metrics[cat]["tpr5"] / n_tpr if n_tpr else float("nan")
            if cat in ["shuffled", "negatives"]:
                results[f"{cat}_fpr1"] = cat_metrics[cat]["fpr1"] / n_fpr if n_fpr else float("nan")

        n_loss = agg_metrics["n_loss"]
        n_tpr = agg_metrics["n_tpr"]
        n_fpr = agg_metrics["n_fpr"]
        results["aggregate_loss"] = agg_metrics["loss"] / n_loss if n_loss else float("nan")
        results["aggregate_tpr1"] = agg_metrics["tpr1"] / n_tpr if n_tpr else float("nan")
        results["aggregate_tpr5"] = agg_metrics["tpr5"] / n_tpr if n_tpr else float("nan")
        results["aggregate_fpr1"] = agg_metrics["fpr1"] / n_fpr if n_fpr else float("nan")

        import numpy as _np
        for cat in ["negatives", "shuffled"]:
            vals = pnone_medians[cat]
            results[f"{cat}_median_pnone"] = float(_np.median(vals)) if vals else float("nan")

        results["aggregate_median_pnone"] = float(_np.median(pnone_all)) if pnone_all else float("nan")
        results["prefilter_accuracy"] = (
            included_coarse_labels / total_coarse_label_sets
            if total_coarse_label_sets > 0
            else float("nan")
        )

        if self.args.local_rank in [-1, 0]:
            print("\n=== Debug: Sequence Category Distribution ===")
            for cat in categories:
                count = debug_counts[cat]
                samples = debug_sample_ids[cat]
                print(f"{cat}: {count} sequences")
                if samples:
                    print(f"  Sample IDs: {samples}")
                else:
                    print("  No sequences found!")
            print("=" * 50)

        prefixed_results = {f"{metric_key_prefix}_{k}": v for k, v in results.items()}
        for key, value in prefixed_results.items():
            if isinstance(value, torch.Tensor):
                prefixed_results[key] = value.item()

        self.log(prefixed_results)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, prefixed_results)

        return prefixed_results

    def create_optimizer(self):
        if self.optimizer is None:
            opt_cfg = self.args.optimizer if hasattr(self.args, "optimizer") else None
            if opt_cfg is None:
                from psalm.config import get_training_config
                training_config = get_training_config()
                opt_cfg = training_config["optimizer"]
                lr_cfg = training_config["learning_rate"]

            esm_params = []
            classifier_params = []

            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name.startswith("esm_model."):
                        esm_params.append(param)
                    else:
                        classifier_params.append(param)

            if not classifier_params:
                raise ValueError(
                    "No classifier parameters found! Check that the classifier head parameters are properly tagged."
                )

            optimizer_grouped_parameters = []
            if esm_params:
                optimizer_grouped_parameters.append(
                    {
                        "params": esm_params,
                        "lr": float(lr_cfg["esm"]),
                        "betas": (float(opt_cfg["esm"]["beta_1"]), float(opt_cfg["esm"]["beta_2"])),
                        "eps": float(opt_cfg["esm"]["epsilon"]),
                        "weight_decay": float(opt_cfg["esm"]["weight_decay"]),
                    }
                )

            optimizer_grouped_parameters.append(
                {
                    "params": classifier_params,
                    "lr": float(lr_cfg["classifier"]),
                    "betas": (float(opt_cfg["classifier"]["beta_1"]), float(opt_cfg["classifier"]["beta_2"])),
                    "eps": float(opt_cfg["classifier"]["epsilon"]),
                    "weight_decay": float(opt_cfg["classifier"]["weight_decay"]),
                }
            )

            from torch.optim import AdamW
            self.optimizer = AdamW(optimizer_grouped_parameters)

        return self.optimizer

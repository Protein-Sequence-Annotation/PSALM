from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import time
import warnings
from importlib.metadata import PackageNotFoundError, version as pkg_version
from datetime import datetime
from collections import OrderedDict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from Bio import SeqIO
from transformers import AutoConfig, AutoTokenizer

from psalm.config import get_model_config
from psalm.inference.cbm_score import add_cbm_scores
from psalm.inference.decoder import annotate_domains


def _auto_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _resolve_device(device: Optional[str]) -> torch.device:
    if device is None or str(device).strip().lower() == "auto":
        return _auto_device()

    try:
        requested = torch.device(str(device).strip().lower())
    except Exception as exc:
        raise ValueError(
            f"Invalid device '{device}'. Use one of: auto, cpu, mps, cuda, cuda:<index>."
        ) from exc

    if requested.type == "cuda" and not torch.cuda.is_available():
        fallback = _auto_device()
        warnings.warn(
            f"Requested device '{device}' is unavailable; falling back to '{fallback.type}'."
        )
        return fallback

    if requested.type == "mps":
        mps_ok = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        if not mps_ok:
            fallback = _auto_device()
            warnings.warn(
                f"Requested device '{device}' is unavailable; falling back to '{fallback.type}'."
            )
            return fallback

    return requested


def _resolve_model_dir_with_source(model_name: str) -> Tuple[Path, str]:
    candidate = Path(model_name).expanduser()
    if candidate.exists():
        return candidate.resolve(), "local_path"
    repo_root = Path(__file__).resolve().parents[2]
    local_candidate = repo_root / "models" / model_name
    if local_candidate.exists():
        return local_candidate.resolve(), "repo_models_dir"
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError(
            "Model path not found locally and huggingface_hub is not available."
        ) from exc
    allow_patterns = [
        "config.json",
        "model.safetensors",
        "model.bin",
        "label_mapping.json",
        "pfam_label_mapping.json",
        "transitions.json",
        "pfam_seed_transitions.json",
        "score.cbm",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
        "tokenizer.json",
    ]
    try:
        cache_dir = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
            local_files_only=True,
        )
        return Path(cache_dir), "hf_cache"
    except Exception:
        cache_dir = snapshot_download(
            repo_id=model_name,
            allow_patterns=allow_patterns,
        )
        return Path(cache_dir), "hf_download"


def _resolve_model_dir(model_name: str) -> Path:
    path, _ = _resolve_model_dir_with_source(model_name)
    return path


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


class _TeeIO:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, data: str) -> None:
        for stream in self.streams:
            stream.write(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def _load_label_mapping(model_dir: Path) -> dict:
    for name in ("label_mapping.json", "pfam_label_mapping.json"):
        path = model_dir / name
        if path.exists():
            return _load_json(path)
    raise FileNotFoundError("label_mapping.json not found in model bundle.")


def _load_transitions(model_dir: Path) -> dict:
    for name in ("transitions.json", "pfam_seed_transitions.json"):
        path = model_dir / name
        if path.exists():
            raw = _load_json(path)
            return {
                int(i): {int(j): float(lp) for j, lp in row.items()}
                for i, row in raw.items()
            }
    raise FileNotFoundError("transitions.json not found in model bundle.")


def _infer_output_size(label_mapping: dict) -> int:
    max_idx = -1
    for pfam, info in label_mapping.items():
        if isinstance(info, dict):
            max_idx = max(max_idx, int(info["start"]), int(info["middle"]), int(info["stop"]))
        else:
            max_idx = max(max_idx, int(info))
    return max_idx + 1


def _load_state_dict(model_dir: Path) -> dict:
    weights_path = model_dir / "model.safetensors"
    if weights_path.exists():
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise ImportError(
                "safetensors is required to load model.safetensors."
            ) from exc
        return load_file(str(weights_path))
    bin_path = model_dir / "model.bin"
    if bin_path.exists():
        try:
            return torch.load(bin_path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError("model.safetensors or model.bin not found in model bundle.")


def _load_model_state_with_fallback(model: nn.Module, state_dict: dict, context: str) -> None:
    try:
        model.load_state_dict(state_dict)
        return
    except RuntimeError as err:
        msg = str(err)
        known_key = "esm_model.embeddings.position_embeddings.weight"
        if known_key not in msg:
            raise

    incompatible = model.load_state_dict(state_dict, strict=False)
    dropped = [k for k in incompatible.unexpected_keys if k == known_key]
    remaining_unexpected = [k for k in incompatible.unexpected_keys if k != known_key]
    if remaining_unexpected:
        raise RuntimeError(
            f"{context}: unexpected checkpoint keys remain after fallback: {remaining_unexpected}"
        )
    if incompatible.missing_keys:
        warnings.warn(
            f"{context}: missing keys during fallback load: {incompatible.missing_keys}"
        )
    # Intentionally suppress the known dropped-key warning by default
    # to keep CLI startup output clean.
    _ = dropped


def _get_psalm_version() -> str:
    try:
        return pkg_version("protein-sequence-annotation")
    except PackageNotFoundError:
        return "0+local"


def _load_cbm_model(model_dir: Path):
    cbm_path = model_dir / "score.cbm"
    if not cbm_path.exists():
        raise FileNotFoundError("score.cbm not found in model bundle.")
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise ImportError(
            "CatBoost is required for CBM scoring. Install catboost to use scan()."
        ) from exc
    model = CatBoostClassifier()
    model.load_model(str(cbm_path))
    return model


def _build_decoder_resources(label_mapping: dict, transitions: dict):
    none_label = label_mapping.get("None")
    if none_label is None:
        raise KeyError("label_mapping.json missing 'None' label.")

    inverse_label_mapping = {}
    for pfam, info in label_mapping.items():
        if pfam == "None":
            inverse_label_mapping[int(info)] = (pfam, None)
        else:
            inverse_label_mapping[int(info["start"])] = (pfam, "start")
            inverse_label_mapping[int(info["middle"])] = (pfam, "middle")
            inverse_label_mapping[int(info["stop"])] = (pfam, "stop")

    starts_full: List[int] = []
    middles_full: List[int] = []
    stops_full: List[int] = []
    for idx, (pf, role) in inverse_label_mapping.items():
        if role == "start":
            starts_full.append(idx)
        elif role == "middle":
            middles_full.append(idx)
        elif role == "stop":
            stops_full.append(idx)

    role_map = {
        "start": np.array(starts_full, dtype=np.int64),
        "middle": np.array(middles_full, dtype=np.int64),
        "stop": np.array(stops_full, dtype=np.int64),
    }

    E = sum(len(row) for row in transitions.values())
    from_list = np.empty(E, dtype=np.int64)
    to_list = np.empty(E, dtype=np.int64)
    lp_list = np.empty(E, dtype=np.float64)
    idx = 0
    for i, row in transitions.items():
        for j, lp in row.items():
            from_list[idx] = i
            to_list[idx] = j
            lp_list[idx] = lp
            idx += 1

    all_pfams = sorted({pf for pf, _ in inverse_label_mapping.values() if pf != "None"})
    pfam2id = {pf: i for i, pf in enumerate(all_pfams)}
    id_to_pfam = all_pfams

    C = max(inverse_label_mapping.keys()) + 1
    state_pfam_ids_full = np.full(C, -1, dtype=np.int64)
    state_role_ids_full = np.full(C, -1, dtype=np.int8)
    role2id = {"start": 0, "middle": 1, "stop": 2}
    for idx, (pf, role) in inverse_label_mapping.items():
        state_pfam_ids_full[idx] = pfam2id.get(pf, -1)
        state_role_ids_full[idx] = role2id.get(role, -1)

    return {
        "none_label": int(none_label),
        "label_mapping": label_mapping,
        "inverse_label_mapping": inverse_label_mapping,
        "role_map": role_map,
        "from_list": from_list,
        "to_list": to_list,
        "lp_list": lp_list,
        "id_to_pfam": id_to_pfam,
        "state_pfam_ids_full": state_pfam_ids_full,
        "state_role_ids_full": state_role_ids_full,
    }


def _resolve_use_fa(use_fa: bool, device: torch.device) -> bool:
    if not use_fa:
        return False
    if device.type != "cuda":
        warnings.warn("FlashAttention requires CUDA; disabling use_fa.")
        return False
    try:
        import faesm  # noqa: F401
    except ImportError:
        warnings.warn("faesm not installed; disabling use_fa.")
        return False
    return True


class PSALM(nn.Module):
    """
    PSALM integrates an ESM model with a single MLP head for Pfam-family classification.
    Supports both training (config-based) and inference from a model bundle.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        ignore_label: int = -100,
        device: Optional[str] = None,
        use_fa: Optional[bool] = None,
        warmup: Optional[bool] = None,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        super().__init__()
        self.model_name = model_name
        self.ignore_label = ignore_label
        self.model_dir: Optional[Path] = None
        self.model_source: Optional[str] = None
        self.cbm_model = None
        self.use_fa = False
        self.warmup_executed = False
        self.resolved_device: Optional[torch.device] = None
        self.status_callback = status_callback

        if model_name is None:
            cfg = get_model_config()
            model_name_cfg = cfg.model_name
            esm_cfg = AutoConfig.from_pretrained(model_name_cfg)
            if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings is not None:
                esm_cfg.max_position_embeddings = cfg.max_position_embeddings
            use_fa_cfg = bool(getattr(cfg, "use_fa", False))
            device_obj = _resolve_device(device)
            self.resolved_device = device_obj
            self.use_fa = _resolve_use_fa(use_fa_cfg, device_obj)
            if self.use_fa:
                from faesm.esm import FAEsmModel
                esm_cfg.use_fa = True
                self.esm_model = FAEsmModel.from_pretrained(
                    model_name_cfg, config=esm_cfg, add_pooling_layer=False
                )
            else:
                from transformers import EsmModel
                self.esm_model = EsmModel.from_pretrained(
                    model_name_cfg, config=esm_cfg, add_pooling_layer=False
                )
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_cfg)
            families = cfg.output_size
        else:
            self.model_dir, self.model_source = _resolve_model_dir_with_source(model_name)
            if self.status_callback is not None:
                self.status_callback("Model source resolved.")
            device_obj = _resolve_device(device)
            self.resolved_device = device_obj
            esm_cfg = AutoConfig.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            if use_fa is None:
                use_fa = bool(getattr(esm_cfg, "use_fa", False))
            self.use_fa = _resolve_use_fa(bool(use_fa), device_obj)
            if self.use_fa:
                from faesm.esm import FAEsmModel
                esm_cfg.use_fa = True
                self.esm_model = FAEsmModel(esm_cfg, add_pooling_layer=False)
            else:
                from transformers import EsmModel
                self.esm_model = EsmModel(esm_cfg, add_pooling_layer=False)
            self.tokenizer = AutoTokenizer.from_pretrained(
                str(self.model_dir), local_files_only=True
            )
            label_mapping = _load_label_mapping(self.model_dir)
            families = _infer_output_size(label_mapping)

        hidden_dim = self.esm_model.config.hidden_size
        expand_dim = 2 * hidden_dim
        self.classes = families

        self.fc1 = nn.Linear(hidden_dim, expand_dim, bias=True)
        self.ln1 = nn.LayerNorm(expand_dim, self.esm_model.config.layer_norm_eps)
        self.fc2 = nn.Linear(expand_dim, expand_dim, bias=True)
        self.ln2 = nn.LayerNorm(expand_dim, self.esm_model.config.layer_norm_eps)
        self.fc3 = nn.Linear(expand_dim, families, bias=True)
        self.relu = nn.ReLU()

        for lin in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for ln in (self.ln1, self.ln2):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

        if model_name is None:
            if cfg.freeze_esm:
                for p in self.esm_model.parameters():
                    p.requires_grad = False
                print("ESM parameters frozen")
            else:
                print("ESM parameters unfrozen for fine-tuning")

            esm_params = sum(p.numel() for p in self.esm_model.parameters() if p.requires_grad)
            classifier_params = sum(
                p.numel() for name, p in self.named_parameters()
                if p.requires_grad and not name.startswith("esm_model.")
            )
            print(f"Trainable parameters: ESM = {esm_params:,}, Classifier = {classifier_params:,}")
        else:
            state_dict = _load_state_dict(self.model_dir)
            _load_model_state_with_fallback(self, state_dict, context="inference bundle")
            resources = _build_decoder_resources(
                label_mapping, _load_transitions(self.model_dir)
            )
            self.none_label = resources["none_label"]
            self.label_mapping = resources["label_mapping"]
            self.inverse_label_mapping = resources["inverse_label_mapping"]
            self.role_map = resources["role_map"]
            self.from_list = resources["from_list"]
            self.to_list = resources["to_list"]
            self.lp_list = resources["lp_list"]
            self.id_to_pfam = resources["id_to_pfam"]
            self.state_pfam_ids_full = resources["state_pfam_ids_full"]
            self.state_role_ids_full = resources["state_role_ids_full"]
            self.cbm_model = _load_cbm_model(self.model_dir)
            self.to(device_obj)
            if warmup is None:
                warmup = True
            if warmup:
                if self.status_callback is not None:
                    self.status_callback("Warmup started.")
                self._warmup()
                self.warmup_executed = True
                if self.status_callback is not None:
                    self.status_callback("Warmup finished.")

    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        if attention_mask is not None:
            attention_mask = attention_mask.bool()

        out = self.esm_model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        reps = out.last_hidden_state

        x = self.fc1(reps)
        x = self.relu(x)
        x = self.ln1(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.ln2(x)

        logits = self.fc3(x)
        output = {"logits": logits}

        if labels is not None:
            C = logits.size(-1)
            loss_fct = nn.CrossEntropyLoss(ignore_index=self.ignore_label)
            output["loss"] = loss_fct(logits.view(-1, C), labels.view(-1))

        return output

    @classmethod
    def from_checkpoint(cls, checkpoint_dir, device=None, ignore_label=-100):
        model = cls(ignore_label=ignore_label)
        weights_path = cls._resolve_checkpoint_path(checkpoint_dir)
        if weights_path.endswith(".safetensors"):
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "safetensors is required to load .safetensors checkpoints."
                ) from exc
            state_dict = load_file(weights_path)
        else:
            try:
                state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
            except TypeError:
                state_dict = torch.load(weights_path, map_location="cpu")
        _load_model_state_with_fallback(model, state_dict, context="checkpoint load")

        device_obj = _resolve_device(device)
        model = model.to(device_obj)
        return model

    @staticmethod
    def _resolve_checkpoint_path(checkpoint_path):
        if os.path.isfile(checkpoint_path):
            return checkpoint_path
        candidates = ["model.safetensors", "pytorch_model.bin"]
        for name in candidates:
            candidate = os.path.join(checkpoint_path, name)
            if os.path.exists(candidate):
                return candidate
        raise FileNotFoundError(
            f"No checkpoint file found in {checkpoint_path}. "
            "Expected model.safetensors or pytorch_model.bin."
        )

    def infer(self, sequence: str) -> torch.Tensor:
        if not hasattr(self, "tokenizer"):
            raise RuntimeError("Tokenizer not initialized. Load a model bundle first.")
        device = next(self.parameters()).device
        toks = self.tokenizer(sequence, return_tensors="pt").to(device)

        self.eval()
        with torch.no_grad():
            if self.use_fa and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(**toks)
            else:
                out = self(**toks)

        logits = out["logits"]
        logits = logits[:, 1:-1, :]
        logits = logits.squeeze(0)
        return logits.float()

    def batch_infer(
        self,
        sequences: List[str],
        max_batch_size: Optional[int] = None,
    ) -> torch.Tensor:
        device = next(self.parameters()).device
        self.eval()
        toks = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(device)

        with torch.no_grad():
            if self.use_fa and device.type == "cuda":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = self(**toks)
            else:
                out = self(**toks)

            return torch.log_softmax(out["logits"], dim=-1)

    def _warmup(self) -> None:
        warmup_path = Path(__file__).resolve().parent / "inference" / "warmup.fasta"
        if not warmup_path.exists():
            return
        try:
            records = list(SeqIO.parse(str(warmup_path), "fasta"))
        except Exception:
            return
        for idx, rec in enumerate(records, start=1):
            seq_id = rec.id or f"seq{idx}"
            try:
                self._scan_one(
                    sequence=str(rec.seq),
                    seq_id=seq_id,
                    score_thresh=0.0,
                    beam_size=64,
                    prior_mid_to_start=3.4e-5,
                    prior_stop_to_mid=4.817e-3,
                    prior_stop_to_start=5.540e-3,
                    T=1.0,
                    refine_extended=True,
                    verbose=False,
                    print_output=False,
                )
            except Exception:
                continue

    def _scan_one(
        self,
        sequence: str,
        seq_id: str,
        score_thresh: float,
        beam_size: int,
        prior_mid_to_start: float,
        prior_stop_to_mid: float,
        prior_stop_to_start: float,
        T: float,
        refine_extended: bool,
        verbose: bool,
        print_output: bool,
    ) -> List[Tuple[str, int, int, float, float, float, float, str]]:
        if any(
            not hasattr(self, attr)
            for attr in ("inverse_label_mapping", "role_map", "none_label")
        ):
            raise RuntimeError("Decoder resources not initialized in this model.")
        if self.cbm_model is None:
            raise RuntimeError("CBM model not loaded; score.cbm is required for scan().")

        t0 = time.time()
        logits = self.infer(sequence)
        device = logits.device
        if T > 0.0:
            logits = logits / T
        log_probs = torch.log_softmax(logits, dim=-1)
        inf_dt = time.time() - t0
        total_fams = int((log_probs.size(1) - 1) / 3)

        L, C = log_probs.shape
        top_idx = log_probs.argmax(dim=1)
        pos_fams = ((top_idx + 2) // 3) * 3
        pfam_ids = ((torch.arange(C, device=device) + 2) // 3) * 3

        keep_full = (pfam_ids.unsqueeze(0) == pos_fams.unsqueeze(1)).any(dim=0)
        keep_full[self.none_label] = True
        keep_full_cpu = keep_full.cpu().contiguous()

        total_kept = int(keep_full_cpu.sum().item())
        K = (total_kept - 1) // 3

        domains_original_scored: List[Tuple[str, int, int, float, float, float, float, str]] = []
        domains_scored: List[Tuple[str, int, int, float, float, float, float, str]] = []
        best_path = None
        gamma = None
        dec_dt = 0.0
        total_domains = 0
        passed = 0

        sub_idx = torch.where(keep_full)[0]
        if not (sub_idx.numel() <= 1 and (sub_idx == self.none_label).all()):
            sub_emits = log_probs.index_select(1, sub_idx)
            probs = sub_emits.exp()
            row_sum = probs.sum(dim=1, keepdim=True)
            probs = probs / row_sum

            none_sub = int((sub_idx == self.none_label).nonzero(as_tuple=False).item())
            sub_emits_cpu = probs.clamp_min(1e-40).log().to(dtype=torch.float32).cpu().contiguous()

            t1 = time.time()
            domains_original, domains, best_path, _, _, _ = annotate_domains(
                sub_emits_cpu,
                keep_full_cpu,
                sequence,
                self.inverse_label_mapping,
                self.label_mapping,
                self.from_list,
                self.to_list,
                self.lp_list,
                self.role_map,
                self.state_pfam_ids_full,
                self.state_role_ids_full,
                self.id_to_pfam,
                none_sub,
                beam_size=beam_size,
                prior_mid_to_start=prior_mid_to_start,
                prior_stop_to_mid=prior_stop_to_mid,
                prior_stop_to_start=prior_stop_to_start,
                refine_extended=refine_extended,
            )
            dec_dt = time.time() - t1
            gamma = probs

            domains_original_scored = add_cbm_scores(domains_original, self.cbm_model)
            domains_scored = add_cbm_scores(domains, self.cbm_model)

            total_domains = len(domains_scored)
            filtered = []
            for pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status in domains_scored:
                if cbm_score >= score_thresh:
                    filtered.append((pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status))
            domains_scored = filtered
            passed = len(domains_scored)

        def _make_match_line(length: int, status: str) -> str:
            if length <= 0:
                return ""
            no_start = "no start" in status
            no_stop = "no stop" in status
            chars = ["="] * length
            if length == 1:
                if no_start and no_stop:
                    chars[0] = "<"
                elif no_start:
                    chars[0] = "<"
                elif no_stop:
                    chars[0] = ">"
                else:
                    chars[0] = "["
                return "".join(chars)
            chars[0] = "<" if no_start else "["
            chars[-1] = ">" if no_stop else "]"
            return "".join(chars)

        if print_output:
            model_label = self.model_name or getattr(self.esm_model.config, "name_or_path", "training")
            device_label = str(next(self.parameters()).device)
            psalm_ver = _get_psalm_version()
            stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            rows = [
                ("PSALM", psalm_ver),
                ("Date/time", stamp),
                ("Model", model_label),
                ("Device", device_label),
                ("Refinement", "On" if refine_extended else "Off"),
                ("Beam size", str(beam_size)),
                ("Embedding time", f"{inf_dt*1000:.2f} ms"),
                ("Decoding time", f"{dec_dt*1000:.2f} ms"),
                ("Family filter", f"{K}/{total_fams} families passed"),
                ("Score filter", f">={score_thresh:.2f}: {passed}/{total_domains} domains passed"),
            ]
            key_w = max(len(k) for k, _ in rows)
            val_w = max(len(v) for _, v in rows)
            border = "+" + "-" * (key_w + 2) + "+" + "-" * (val_w + 2) + "+"
            print(border)
            for k, v in rows:
                print(f"| {k:<{key_w}} | {v:<{val_w}} |")
            print(border)
            print(f">>> Query: {seq_id} ({len(sequence)} aa)")

            sorted_domains = sorted(domains_scored, key=lambda x: x[3], reverse=True)
            score_w = max(len("Score"), *(len(f"{d[3]:.3f}") for d in sorted_domains)) if sorted_domains else len("Score")
            bit_w = max(len("Bit score"), *(len(f"{d[4]:.2f}") for d in sorted_domains)) if sorted_domains else len("Bit score")
            bias_w = max(len("Bias"), *(len(f"{d[6]:.2f}") for d in sorted_domains)) if sorted_domains else len("Bias")
            len_ratio_w = max(len("Len Frac"), *(len(f"{d[5]:.2f}") for d in sorted_domains)) if sorted_domains else len("Len Frac")
            model_names = [self.label_mapping[pfam]["family_name"] for pfam, *_ in sorted_domains] if sorted_domains else []
            model_w = max(len("Model"), *(len(n) for n in model_names)) if model_names else len("Model")
            desc_w = max(len("Description"), *(len(self.label_mapping[pfam]["family_desc"]) for pfam, *_ in sorted_domains)) if sorted_domains else len("Description")
            pfam_w = max(len("Pfam"), *(len(pfam) for pfam, *_ in sorted_domains)) if sorted_domains else len("Pfam")

            print("--- Pfam domain hits ---")
            header_fmt = (
                f"{{:>{score_w}}}   "
                f"{{:<{model_w}}}   {{:<{desc_w}}}   {{:<{pfam_w}}}   "
                f"{{:>5}}   {{:>5}}"
            )
            print(header_fmt.format("Score", "Model", "Description", "Pfam", "Start", "Stop"))
            print("-" * (score_w+3+model_w+3+desc_w+3+pfam_w+3+5+3+5))

            for pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status in sorted_domains:
                name = self.label_mapping[pfam]["family_name"]
                desc_f = self.label_mapping[pfam]["family_desc"]
                print(
                    f"{cbm_score:>{score_w}.3f}   "
                    f"{name:<{model_w}}   {desc_f:<{desc_w}}   {pfam:<{pfam_w}}   "
                    f"{start:>5}   {stop:>5}"
                )
            print()

            if verbose and domains_scored:
                family_hits = OrderedDict()
                for pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status in domains_scored:
                    family_hits.setdefault(pfam, []).append((start, stop, cbm_score, bit_score, len_ratio, bias, status))

                print("Domain annotation for each model (and alignments):\n")

                nested_components = {}
                for pfam, start_combined, stop_combined, cbm_score, bit_score, len_ratio, bias, status in domains_scored:
                    if status == "full (merged)":
                        components = []
                        for pfam_orig, start_orig, stop_orig, cbm_o, bit_o, len_o, bias_o, status_o in domains_original_scored:
                            if (
                                pfam_orig == pfam
                                and start_orig >= start_combined
                                and stop_orig <= stop_combined
                                and status_o in [
                                    "partial (no stop)",
                                    "partial (no start)",
                                    "partial (no start or stop)",
                                ]
                            ):
                                components.append((start_orig, stop_orig, cbm_o, bit_o, len_o, bias_o, status_o))
                        if len(components) >= 2:
                            components.sort(key=lambda x: x[0])
                            nested_components[(pfam, start_combined, stop_combined)] = components

                for pfam, hits in family_hits.items():
                    name = self.label_mapping[pfam]["family_name"]
                    desc_f = self.label_mapping[pfam]["family_desc"]
                    print(f">> {name}   {desc_f}   ({pfam})\n")

                    idx_w = max(len("#"), len(str(len(hits))))
                    hdr = (
                        f"{{:>{idx_w}}}   {{:>{score_w}}}   {{:>{bit_w}}}   {{:>{bias_w}}}   {{:>{len_ratio_w}}}   "
                        f"{{:>5}}   {{:>5}}   {{:<10}}   {{:<{pfam_w}}}"
                    )
                    print(hdr.format("#", "Score", "Bit score", "Bias", "Len Frac", "Start", "Stop", "Status", "Pfam"))
                    print("-" * (idx_w+3+score_w+3+bit_w+3+bias_w+3+len_ratio_w+3+5+3+5+3+10+3+pfam_w))

                    for i, (start, stop, cbm_score, bit_score, len_ratio, bias, status) in enumerate(hits, start=1):
                        print(
                            f"{i:>{idx_w}}   "
                            f"{cbm_score:>{score_w}.3f}   "
                        f"{bit_score:>{bit_w}.2f}   "
                        f"{bias:>{bias_w}.2f}   "
                        f"{len_ratio:>{len_ratio_w}.2f}   "
                            f"{start:>5}   "
                            f"{stop:>5}   "
                            f"{status:<10}   "
                            f"{pfam}"
                        )
                    print()

                    indent = " " * 6
                    for i, (start, stop, cbm_score, bit_score, len_ratio, bias, status) in enumerate(hits, start=1):
                        nested_key = (pfam, start, stop)
                        if nested_key in nested_components:
                            components = nested_components[nested_key]
                            print(
                                f"== domain {i}    score: {cbm_score:.3f}    bit_score: {bit_score:.2f}    "
                                f"bias: {bias:.2f}    len_ratio: {len_ratio:.2f}    [nested]"
                            )

                            for comp_idx, (comp_start, comp_stop, cbm_c, bit_c, len_c, bias_c, status_c) in enumerate(components, 1):
                                print(
                                    f"   component {comp_idx}    score: {cbm_c:.3f}    bit_score: {bit_c:.2f}    "
                                    f"bias: {bias_c:.2f}    len_ratio: {len_c:.2f}    {status_c}"
                                )
                                region_seq = sequence[comp_start-1:comp_stop]
                                length = len(region_seq)

                                match_str = _make_match_line(length, status_c)
                                prob_chars = []
                                for offset in range(length):
                                    t = comp_start - 1 + offset
                                    p = float(gamma[t, best_path[t]]) if gamma is not None else 0.0
                                    if p < 0.05:
                                        pc = "0"
                                    elif p < 0.95:
                                        pc = str(min(9, int(p * 10)))
                                    else:
                                        pc = "*"
                                    prob_chars.append(pc)
                                prob_str = "".join(prob_chars)

                                for off in range(0, length, 80):
                                    seq_chunk = region_seq[off:off+80]
                                    m_chunk = match_str[off:off+80]
                                    p_chunk = prob_str[off:off+80]
                                    abs_s = comp_start + off
                                    abs_e = comp_start + off + len(seq_chunk) - 1
                                    print(f"{abs_s:>5} {seq_chunk:<80} {abs_e:>5}")
                                    print(indent + m_chunk)
                                    print(indent + p_chunk)
                                print()
                        else:
                            print(
                                f"== domain {i}    score: {cbm_score:.3f}    bit_score: {bit_score:.2f}    "
                                f"bias: {bias:.2f}    len_ratio: {len_ratio:.2f}"
                            )
                            region_seq = sequence[start-1:stop]
                            length = len(region_seq)

                            match_str = _make_match_line(length, status)
                            prob_chars = []
                            for offset in range(length):
                                t = start - 1 + offset
                                p = float(gamma[t, best_path[t]]) if gamma is not None else 0.0
                                if p < 0.05:
                                    pc = "0"
                                elif p < 0.95:
                                    pc = str(min(9, int(p * 10)))
                                else:
                                    pc = "*"
                                prob_chars.append(pc)
                            prob_str = "".join(prob_chars)

                            for off in range(0, length, 80):
                                seq_chunk = region_seq[off:off+80]
                                m_chunk = match_str[off:off+80]
                                p_chunk = prob_str[off:off+80]
                                abs_s = start + off
                                abs_e = start + off + len(seq_chunk) - 1
                                print(f"{abs_s:>5} {seq_chunk:<80} {abs_e:>5}")
                                print(indent + m_chunk)
                                print(indent + p_chunk)
                            print()
                    print()

        return domains_scored

    def scan(
        self,
        sequence: str = None,
        fasta: str = None,
        score_thresh: float = 0.0,
        beam_size: int = 64,
        prior_mid_to_start: float = 3.4e-5,
        prior_stop_to_mid: float = 4.817e-3,
        prior_stop_to_start: float = 5.540e-3,
        T: float = 1.0,
        refine_extended: bool = True,
        to_tsv: str = None,
        to_txt: str = None,
        verbose: bool = False,
        _seq_id: Optional[str] = None,
        _print_output: bool = True,
    ) -> Dict[str, List[Tuple[str, int, int, float, float, float, float, str]]]:
        """
        Scan a protein sequence for domain annotations and display results.

        Args:
            sequence: Protein sequence as a string (or use fasta instead)
            fasta: Path to FASTA file containing one or more protein sequences
            score_thresh: CBM score threshold for filtering domains
            beam_size: Beam size for decoding
            prior_mid_to_start: Prior for middle to start transition
            prior_stop_to_mid: Prior for stop to middle transition
            prior_stop_to_start: Prior for stop to start transition
            T: Temperature scaling factor for logits (1.0 = no scaling)
            refine_extended: Run extended refinement for long domains
            to_tsv: Path to save results as TSV (optional)
            to_txt: Path to save printed output as text (optional)
            verbose: If True, print detailed results and alignments

        Returns:
            Dict mapping seq_id -> list of domain tuples:
              (pfam, start, stop, cbm_score, bit_score, len_ratio, bias, status)
        """
        def _normalize_path(path: Optional[str], default_ext: str) -> Optional[str]:
            if path is None:
                return None
            base, ext = os.path.splitext(path)
            return path if ext else f"{base}{default_ext}"

        txt_path = _normalize_path(to_txt, ".txt")
        emit_console = bool(_print_output)
        emit_text_file = bool(txt_path)
        emit_printable = emit_console or emit_text_file

        if _seq_id is None and fasta is not None:
            records = list(SeqIO.parse(fasta, "fasta"))
            if len(records) > 1:
                results: Dict[str, List[Tuple[str, int, int, float, float, float, float, str]]] = {}
                used_ids: Dict[str, int] = {}
                unnamed = 0
                buffer = None
                output_context = contextlib.nullcontext()
                if emit_text_file and emit_console:
                    buffer = io.StringIO()
                    tee = _TeeIO(sys.stdout, buffer)
                    output_context = contextlib.redirect_stdout(tee)
                elif emit_text_file and not emit_console:
                    buffer = io.StringIO()
                    output_context = contextlib.redirect_stdout(buffer)
                with output_context:
                    for rec in records:
                        rec_id = (rec.id or "").strip()
                        if rec_id == "":
                            unnamed += 1
                            rec_id = f"seq{unnamed}"
                        if rec_id in used_ids:
                            used_ids[rec_id] += 1
                            rec_id = f"{rec_id}_{used_ids[rec_id]}"
                        else:
                            used_ids[rec_id] = 1
                        sub = self.scan(
                            sequence=str(rec.seq),
                            score_thresh=score_thresh,
                            beam_size=beam_size,
                            prior_mid_to_start=prior_mid_to_start,
                            prior_stop_to_mid=prior_stop_to_mid,
                            prior_stop_to_start=prior_stop_to_start,
                            T=T,
                            refine_extended=refine_extended,
                            to_tsv=None,
                            to_txt=None,
                            verbose=verbose,
                            _seq_id=rec_id,
                            _print_output=emit_printable,
                        )
                        results.update(sub)
                        if emit_printable:
                            print()
                if emit_text_file and buffer is not None:
                    with open(txt_path, "w", encoding="utf-8") as handle:
                        handle.write(buffer.getvalue())
                if to_tsv:
                    rows = []
                    for seq_id, domains in results.items():
                        for domain in domains:
                            rows.append((seq_id, *domain))
                    df = pd.DataFrame(
                        rows,
                        columns=["seq_id", "pfam", "start", "stop", "score", "bit_score", "len_ratio", "bias", "status"],
                    )
                    df.to_csv(to_tsv, sep="\t", index=False)
                return results

        if fasta is not None:
            rec = SeqIO.read(fasta, "fasta")
            sequence = str(rec.seq)
            seq_id = (rec.id or "").strip()
        else:
            seq_id = ""
        if _seq_id is not None:
            seq_id = _seq_id
        if seq_id == "":
            seq_id = "seq1"

        if sequence is None:
            raise ValueError("Provide either sequence or fasta.")
        if emit_text_file and emit_console:
            buffer = io.StringIO()
            tee = _TeeIO(sys.stdout, buffer)
            with contextlib.redirect_stdout(tee):
                domains_scored = self._scan_one(
                    sequence=sequence,
                    seq_id=seq_id,
                    score_thresh=score_thresh,
                    beam_size=beam_size,
                    prior_mid_to_start=prior_mid_to_start,
                    prior_stop_to_mid=prior_stop_to_mid,
                    prior_stop_to_start=prior_stop_to_start,
                    T=T,
                    refine_extended=refine_extended,
                    verbose=verbose,
                    print_output=emit_printable,
                )
            with open(txt_path, "w", encoding="utf-8") as handle:
                handle.write(buffer.getvalue())
        elif emit_text_file and not emit_console:
            buffer = io.StringIO()
            with contextlib.redirect_stdout(buffer):
                domains_scored = self._scan_one(
                    sequence=sequence,
                    seq_id=seq_id,
                    score_thresh=score_thresh,
                    beam_size=beam_size,
                    prior_mid_to_start=prior_mid_to_start,
                    prior_stop_to_mid=prior_stop_to_mid,
                    prior_stop_to_start=prior_stop_to_start,
                    T=T,
                    refine_extended=refine_extended,
                    verbose=verbose,
                    print_output=emit_printable,
                )
            with open(txt_path, "w", encoding="utf-8") as handle:
                handle.write(buffer.getvalue())
        else:
            domains_scored = self._scan_one(
                sequence=sequence,
                seq_id=seq_id,
                score_thresh=score_thresh,
                beam_size=beam_size,
                prior_mid_to_start=prior_mid_to_start,
                prior_stop_to_mid=prior_stop_to_mid,
                prior_stop_to_start=prior_stop_to_start,
                T=T,
                refine_extended=refine_extended,
                verbose=verbose,
                print_output=emit_printable,
            )

        if to_tsv:
            df = pd.DataFrame(
                domains_scored,
                columns=["pfam", "start", "stop", "score", "bit_score", "len_ratio", "bias", "status"],
            )
            df.insert(0, "seq_id", seq_id)
            df.to_csv(to_tsv, sep="\t", index=False)

        return {seq_id: domains_scored}

    def decode_sequence(self, *args, **kwargs) -> Dict[str, List[Tuple[str, int, int, float, float, float, float, str]]]:
        warnings.warn(
            "decode_sequence() is deprecated and will be removed in a future version. "
            "Use scan() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.scan(*args, **kwargs)

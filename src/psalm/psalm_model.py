import os
import torch
import torch.nn as nn
from transformers import AutoConfig

from psalm.config import get_model_config


class PSALM(nn.Module):
    """
    PSALM integrates an ESM model with a single MLP head for Pfam-family classification.
    Training-only variant (no scan/decoder/inference utilities).
    """

    def __init__(self, ignore_label=-100):
        super().__init__()
        cfg = get_model_config()
        model_name = cfg.model_name
        self.ignore_label = ignore_label

        esm_cfg = AutoConfig.from_pretrained(model_name)
        if hasattr(cfg, "max_position_embeddings") and cfg.max_position_embeddings is not None:
            esm_cfg.max_position_embeddings = cfg.max_position_embeddings

        if cfg.use_fa:
            try:
                from faesm.esm import FAEsmModel
            except ImportError as exc:
                raise ImportError(
                    "use_fa is enabled but faesm is not installed."
                ) from exc
            esm_cfg.use_fa = cfg.use_fa
            self.esm_model = FAEsmModel.from_pretrained(
                model_name, config=esm_cfg, add_pooling_layer=False
            )
        else:
            from transformers import EsmModel
            self.esm_model = EsmModel.from_pretrained(
                model_name, config=esm_cfg, add_pooling_layer=False
            )

        hidden_dim = self.esm_model.config.hidden_size
        expand_dim = 2 * hidden_dim
        families = cfg.output_size
        self.classes = families

        self.fc1 = nn.Linear(hidden_dim, expand_dim, bias=True)
        self.ln1 = nn.LayerNorm(expand_dim, esm_cfg.layer_norm_eps)
        self.fc2 = nn.Linear(expand_dim, expand_dim, bias=True)
        self.ln2 = nn.LayerNorm(expand_dim, esm_cfg.layer_norm_eps)
        self.fc3 = nn.Linear(expand_dim, families, bias=True)
        self.relu = nn.ReLU()

        for lin in (self.fc1, self.fc2, self.fc3):
            nn.init.xavier_uniform_(lin.weight)
            nn.init.zeros_(lin.bias)
        for ln in (self.ln1, self.ln2):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)

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
        model.load_state_dict(state_dict)

        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
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

import os
import yaml
from transformers import EsmConfig

_config = None


def load_config(config_path):
    if config_path is None:
        raise ValueError("config_path must be provided.")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_config(config_path):
    global _config
    _config = load_config(config_path)


def set_config_dict(config_dict):
    global _config
    _config = config_dict


def _ensure_config_loaded():
    global _config
    if _config is None:
        raise RuntimeError(
            "Config is not loaded. Call set_config(path) or set_config_dict(...) first."
        )


def get_model_config():
    """
    Returns a Hugging Face EsmConfig object built from the model parameters in
    the YAML config, with additional custom parameters added as attributes.
    """
    _ensure_config_loaded()
    model_config = _config["model"]

    esm_config = EsmConfig.from_pretrained(model_config["model_name"])

    custom_params = {
        "pretrained_checkpoint_path",
        "freeze_esm",
        "use_fa",
        "output_size",
        "max_batch_size",
        "max_position_embeddings",
        "label_mapping_path",
    }

    for key, value in model_config.items():
        if key in custom_params or not hasattr(esm_config, key):
            setattr(esm_config, key, value)

    return esm_config


def get_training_config():
    """
    Returns the training configuration as a dictionary from the YAML config.
    """
    _ensure_config_loaded()
    return _config["training"]


def get_data_config():
    """
    Returns the data processing configuration as a dictionary from the YAML config.
    """
    _ensure_config_loaded()
    return _config["data"]

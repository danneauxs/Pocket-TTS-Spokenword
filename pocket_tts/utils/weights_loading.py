from pathlib import Path

import safetensors


def get_flow_lm_state_dict(path: Path) -> dict:
    """Load a flow language model state dictionary from a file.
    Args:
    path (Path): The path to the state dictionary file.
    Returns:
    dict: A dictionary containing the loaded model weights, excluding lookup table weights and specific conditioner weights.
    """
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if (
                key.startswith("flow.w_s_t.")
                or key == "condition_provider.conditioners.transcript_in_segment.learnt_padding"
                or key == "condition_provider.conditioners.speaker_wavs.learnt_padding"
            ):
                # skip lookup table weights
                continue
            new_name = key
            if key == "condition_provider.conditioners.transcript_in_segment.embed.weight":
                new_name = "conditioner.embed.weight"
            if key == "condition_provider.conditioners.speaker_wavs.output_proj.weight":
                new_name = "speaker_proj_weight"
            state_dict[new_name] = f.get_tensor(key)
    return state_dict


def get_mimi_state_dict(path: Path) -> dict:
    """Returns a dictionary of model states from a given path, excluding specific VQ keys.
    Args:
    - path (Path): The file path to the model state dictionary.
    """
    state_dict = {}
    with safetensors.safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if key.startswith("model.quantizer.vq.") or key == "model.quantizer.logvar_proj.weight":
                # skip vq weights
                continue

            state_dict[key.removeprefix("model.")] = f.get_tensor(key)
    return state_dict

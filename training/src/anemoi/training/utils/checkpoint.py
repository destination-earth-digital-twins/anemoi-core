# (C) Copyright 2024 Anemoi contributors.
#
# This software is licensed under the terms of the Apache Licence Version 2.0
# which can be obtained at http://www.apache.org/licenses/LICENSE-2.0.
#
# In applying this licence, ECMWF does not waive the privileges and immunities
# granted to it by virtue of its status as an intergovernmental organisation
# nor does it submit to any jurisdiction.

from __future__ import annotations

import re
import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist

from anemoi.training.train.forecaster import GraphForecaster
from anemoi.utils.checkpoints import save_metadata

LOGGER = logging.getLogger(__name__)


def load_and_prepare_model(lightning_checkpoint_path: str) -> tuple[torch.nn.Module, dict]:
    """Load the lightning checkpoint and extract the pytorch model and its metadata.

    Parameters
    ----------
    lightning_checkpoint_path : str
        path to lightning checkpoint

    Returns
    -------
    tuple[torch.nn.Module, dict]
        pytorch model, metadata

    """
    module = GraphForecaster.load_from_checkpoint(lightning_checkpoint_path)
    model = module.model

    metadata = dict(**model.metadata)
    model.metadata = None
    model.config = None

    return model, metadata


def save_inference_checkpoint(model: torch.nn.Module, metadata: dict, save_path: Path | str) -> Path:
    """Save a pytorch checkpoint for inference with the model metadata.

    Parameters
    ----------
    model : torch.nn.Module
        Pytorch model
    metadata : dict
        Anemoi Metadata to inject into checkpoint
    save_path : Path | str
        Directory to save anemoi checkpoint

    Returns
    -------
    Path
        Path to saved checkpoint
    """
    save_path = Path(save_path)
    inference_filepath = save_path.parent / f"inference-{save_path.name}"

    torch.save(model, inference_filepath)
    save_metadata(inference_filepath, metadata)
    return inference_filepath


#def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
#
#    # Load the checkpoint
#    checkpoint = torch.load(ckpt_path, map_location=model.device)
#
#    # Filter out layers with size mismatch
#    state_dict = checkpoint["state_dict"]
#
#    model_state_dict = model.state_dict()
#
#    for key in state_dict.copy():
#        if key in model_state_dict and state_dict[key].shape != model_state_dict[key].shape:
#            LOGGER.info("Skipping loading parameter: %s", key)
#            LOGGER.info("Checkpoint shape: %s", str(state_dict[key].shape))
#            LOGGER.info("Model shape: %s", str(model_state_dict[key].shape))
#
#            del state_dict[key]  # Remove the mismatched key
#
#   # Load the filtered st-ate_dict into the model
#    model.load_state_dict(state_dict, strict=False)
#    return model
# def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
#     # Load the checkpoint
#     checkpoint = torch.load(ckpt_path, map_location=model.device)

#     # Filter out layers with size mismatch
#     state_dict = checkpoint["state_dict"].copy()
#     model_state_dict = model.state_dict()
#     mapping={
#         "layer_norm_attention_src": "layer_norm1",
#         "layer_norm_attention_dest": "layer_norm2",
#         "layer_norm_attention": "layer_norm1",
#         "layer_norm_mlp" : "layer_norm2",
#     }

#     """for state_key in model_state_dict:
#         if state_key not in state_dict:
#             print(f"key: {state_key} is not present in checkpoint.")
#             old_key = state_key

#             for new, old in mapping.items():
#                 if old_key.find(new) != -1:
#                     print(new, old,old_key, old_key.find(new))

#                     print(new, state_key)
#                     old_key = old_key.replace(new, old)
#                     print(f"Performing mapping between (old) {old_key} -> {state_key} (new)")


#             if old_key != state_key:
#                 shape_state_dict = state_dict[old_key].shape
#                 shape_model_state_dict = model_state_dict[state_key].shape

#                 print(shape_state_dict, shape_model_state_dict)
#                 print("Checking shape...")
#                 if shape_state_dict == shape_model_state_dict:
#                     state_dict[state_key] = state_dict.pop(old_key)
#                     print(f"replaced {old_key} with {state_key} into checkpoint... ")
#                 else:
#                     print(f"Warning! Shape missmatch: ckpt: {shape_state_dict} | model: {shape_model_state_dict}")
#                     del state_dict[old_key] # remove state
#     """
#     if torch.dist.get_rank()==0:
#     import re
#     for state_key in list(model_state_dict.keys()):
#         if state_key not in list(state_dict.keys()):
#             print(f"key: {state_key} is not present in checkpoint.")
#             new_key = state_key

#             for new, old in mapping.items():
#                 old_key = re.sub(rf'\b{re.escape(new)}\b', old, new_key)
#                 if old_key in list(state_dict.keys()):
#                     #old_key = old_key.replace(new, old)

#                     shape_state_dict = state_dict[old_key].shape
#                     shape_model_state_dict = model_state_dict[state_key].shape

#                     print(shape_state_dict, shape_model_state_dict)
#                     print("Checking shape...")
#                     if shape_state_dict == shape_model_state_dict:
#                         print(f"Replacing keyname {old} -> {new} in: {old_key}")
#                         state_dict[state_key] = state_dict.pop(old_key)
#                     else:
#                         print(f"Warning! Shape mismatch: {new_key}: {state_dict[new_key].shape} | {state_key}: {model_state_dict[state_key].shape}")
#                         print(f"Removing key: {old_key} from state_dict")
#                         del state_dict[old_key]



#             # print(f"Final mapping: {old_key} -> {state_key}")

#             # if old_key not in list(state_dict.keys()):
#             #     print(f"Error: {old_key} not found in state_dict! Continuing search")
#             #     continue  # Skip instead of crashing

#             # # Shape checking
#             # shape_state_dict = state_dict[old_key].shape
#             # shape_model_state_dict = model_state_dict[state_key].shape
#             # if shape_state_dict == shape_model_state_dict:
#             #     state_dict[state_key] = state_dict.pop(old_key)
#             #     print(f"Replaced {old_key} with {state_key} in checkpoint.")
#             # else:
#             #     print(f"Warning! Shape mismatch: ckpt: {shape_state_dict} | model: {shape_model_state_dict}")
#             #     if old_key in state_dict:
#             #         del state_dict[old_key]  # Safe deletion

#     # Load the filtered st-ate_dict into the model
#     model.load_state_dict(state_dict, strict=False)
#     return model

def transfer_learning_loading(model: torch.nn.Module, ckpt_path: Path | str) -> nn.Module:
    # Load the checkpoint
    print("CHECKPOINT PATH ", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=model.device)

    # Filter out layers with size mismatch
    state_dict = checkpoint["state_dict"].copy()
    model_state_dict = model.state_dict()

    mapping = {
        "layer_norm_attention": "layer_norm1",
        "layer_norm_mlp": "layer_norm2",
        "layer_norm_attention_src": "layer_norm1",
        "layer_norm_attention_dest": "layer_norm2",
    }

    for state_key in list(model_state_dict.keys()):
        if state_key not in state_dict:
            print(f"key: {state_key} is not present in checkpoint.")
            new_key = state_key

            for new, old in mapping.items():
                print("NEW", new)
                print("OLD", old)
                old_key = re.sub(rf'\b{re.escape(new)}\b', old, new_key)
                print("old_key ", old_key)
                if old_key in state_dict:
                    print(f"old key {old_key} is in state dict")
                    shape_state_dict = state_dict[old_key].shape
                    shape_model_state_dict = model_state_dict[state_key].shape

                    print("Checking shape...")
                    if shape_state_dict == shape_model_state_dict:
                        print(f"Replacing keyname {old} -> {new} in: {old_key}")
                        state_dict[state_key] = state_dict[old_key]
                    else:
                        print(f"Warning! Shape mismatch: {old_key}: {state_dict[old_key].shape} | {state_key}: {model_state_dict[state_key].shape}")
                        print(f"Removing key: {old_key} from state_dict")
                        del state_dict[old_key]
                else:
                    print(f"old key {old_key} not in state dict")
        assert (state_key in state_dict), (f"state key {state_key} not in checkpoint")

    # Load the filtered state_dict into the model
    model.load_state_dict(state_dict, strict=False)
    
    return model

def freeze_submodule_by_name(module: nn.Module, target_name: str) -> None:
    """
    Recursively freezes the parameters of a submodule with the specified name.

    Parameters
    ----------
    module : torch.nn.Module
        Pytorch model
    target_name : str
        The name of the submodule to freeze.
    """
    for name, child in module.named_children():
        # If this is the target submodule, freeze its parameters
        if name == target_name:
            for param in child.parameters():
                param.requires_grad = False
        else:
            # Recursively search within children
            freeze_submodule_by_name(child, target_name)



"""model.model.encoder.proc.layer_norm_attention.weight 

model.model.encoder.proc.layer_norm_mlp.weight

for key in model_state_dict.keys():
    print("KEY ", key)
    print("Key in checkpoint? : %s", (key in state_dict)) -> False
    if "layer_norm_attention_src" in key:
        state_dict[key] = state_dict[key.replace("layer_norm_attention_src", "layer_norm1")]
    elif "layer_norm_attention_dest" in key:
        state_dict[key] = state_dict[key.replace("layer_norm_attention_dest", "layer_norm2")]
    elif "layer_norm_attention" in key:
        state_dict[key] = state_dict[key.replace("layer_norm_attention", "layer_norm1")]
    elif "layer_norm_mlp" in key:
        state_dict[key] = state_dict[key.replace("layer_norm_mlp", "layer_norm2")]"""
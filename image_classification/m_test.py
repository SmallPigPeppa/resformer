from timm import create_model
from models import ResFormer
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


# def load_timm_pretrained_weights(model, model_name, checkpoint_path=None):
#     # If a path is provided, load specific checkpoint, otherwise load default pretrained weights
#     if checkpoint_path:
#         state_dict = torch.load(checkpoint_path)
#     else:
#         # Create a model with pretrained weights from 'timm'
#         timm_model = create_model(model_name, pretrained=True)
#         state_dict = timm_model.state_dict()
#
#     # Adapt timm pretrained model keys to match the expected keys in your model
#     new_state_dict = OrderedDict()
#     for k, v in state_dict.items():
#         # Here you would modify the key names to match those expected by your model
#         # This depends on how your model's state_dict keys are named
#         name = k.replace('module.', '')  # remove 'module.' of dataparallel
#         new_state_dict[name] = v
#
#     # Load the adapted state_dict
#     model.load_state_dict(new_state_dict, strict=True)
def load_timm_pretrained_weights(model, model_name, checkpoint_path=None):
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        print("Loaded weights from specified checkpoint.")
    else:
        # Create a model with pretrained weights from 'timm'
        timm_model = create_model(model_name, pretrained=True)
        state_dict = timm_model.state_dict()
        print(f"Loaded default pretrained weights for {model_name}.")

    # Adapt timm pretrained model keys to match the expected keys in your model
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # Remove 'module.' prefix if using DataParallel
        new_state_dict[name] = v

    model_state_dict = model.state_dict()
    successful_loads = []
    failed_loads = []

    for name, param in new_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                model_state_dict[name].copy_(param)
                successful_loads.append(name)
            else:
                failed_loads.append((name, "Mismatched tensor shape"))
        else:
            failed_loads.append((name, "Key not found in model"))

    # Report successfully loaded weights
    if successful_loads:
        print("Successfully loaded the following weights:")
        for name in successful_loads:
            print(f"  {name}")

    # Report weights that failed to load
    if failed_loads:
        print("Failed to load the following weights:")
        for name, reason in failed_loads:
            print(f"  {name}: {reason}")

    return model



# Example Usage:
# model_name must match one of the timm model names that corresponds to your model architecture
# model_name = 'resformer_base_patch16'
model_name = 'deit_base_distilled_patch16_224.fb_in1k'

model = ResFormer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
load_timm_pretrained_weights(model, model_name)

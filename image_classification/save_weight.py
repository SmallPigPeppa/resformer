from timm import create_model
from models import ResFormer
import torch
import torch.nn as nn
from functools import partial
from collections import OrderedDict


def load_timm_pretrained_weights(model, model_name, checkpoint_path=None, save_path=None):
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path)
        print("Loaded weights from specified checkpoint.")
    else:
        # Create a model with pretrained weights from 'timm'
        timm_model = create_model(model_name, pretrained=True)
        state_dict = timm_model.state_dict()
        print(f"Loaded default pretrained weights for {model_name}.")
        if save_path:
            # Save the pretrained weights to a specified path
            torch.save(state_dict, save_path)
            print(f"Saved pretrained weights to {save_path}")

    # Adapt timm pretrained model keys to match the expected keys in your model
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '')  # Remove 'module.' prefix if using DataParallel
        new_state_dict[name] = v

    adapted_state_dict = OrderedDict()
    model_state_dict = model.state_dict()
    successful_loads = []
    failed_loads = []

    for name, param in new_state_dict.items():
        if name in model_state_dict:
            if model_state_dict[name].shape == param.shape:
                adapted_state_dict[name] = param
                successful_loads.append(name)
            else:
                failed_loads.append((name, "Mismatched tensor shape"))
        else:
            failed_loads.append((name, "Key not found in model"))

    # Report successfully adapted weights
    if successful_loads:
        print("The following weights are adapted and ready to load:")
        for name in successful_loads:
            print(f"  {name}")

    # Report weights that could not be adapted
    if failed_loads:
        print("Failed to adapt the following weights:")
        for name, reason in failed_loads:
            print(f"  {name}: {reason}")

    return adapted_state_dict


model_name = 'deit_base_distilled_patch16_224.fb_in1k'

model = ResFormer(patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4, qkv_bias=True,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))
a = load_timm_pretrained_weights(model, model_name, save_path='deit_base_distilled_patch16_224.fb_in1k.pth')

a = load_timm_pretrained_weights(model, model_name, checkpoint_path='deit_base_distilled_patch16_224.fb_in1k.pth')

model.load_state_dict(a, strict=True)

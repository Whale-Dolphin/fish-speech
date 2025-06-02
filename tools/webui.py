import gradio as gr
import torch

from fish_speech.models.text2semantic.inference import (
    load_model,
    generate_long
)
from fish_speech.models.dac.modded_dac import DAC

def filter_state_dict_shapes(params, model):
    model_state_dict = model.state_dict()
    filtered_state_dict = {
        k: v
        for k, v in params.items()
        if k in model_state_dict and v.shape == model_state_dict[k].shape
    }
    skipped_keys = set(params.keys()) - set(filtered_state_dict.keys())
    if skipped_keys:
        print(f"Warning: Skipped loading some keys due to shape mismatch: {skipped_keys}")
        return filtered_state_dict, skipped_keys

def decode_vq(model: DAC, vq_seq: torch.Tensor, vq_length: torch.Tensor):
    

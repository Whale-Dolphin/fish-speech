from fish_speech.models.dac.modded_dac import DAC
import hydra
import librosa
import soundfile as sf
import torch
import numpy as np
from omegaconf import OmegaConf
from typing import Optional, Union

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

def load_dac_model():
    model = hydra.utils.instantiate(OmegaConf.load("fish_speech/configs/modded_dac_vq.yaml"))
    sd = torch.load('checkpoints/openaudio-s1-mini/firefly-gan-large.pth', map_location='cpu')
    filtered_sd, skipped_keys = filter_state_dict_shapes(sd, model)
    print(f"Skipped keys: {skipped_keys}")
    model.load_state_dict(filtered_sd, strict=False)
    model.eval()
    return model

def wav_to_vq(wav_path: str, save_vq: bool = True, model: Optional[DAC] = None):

    if model is None:
        model = load_dac_model()
    
    wave_np, _ = librosa.load(wav_path, sr=44100, mono=False)
    wave_tensor = torch.from_numpy(wave_np).unsqueeze(1)

    with torch.no_grad():
        indices, indices_lengths = model.encode(wave_tensor)
        vq_codes = indices.cpu().numpy()
        vq_lengths = indices_lengths.cpu().numpy()

    if save_vq:
        vq_batch = {
            "vq_codes": vq_codes,
            "vq_lengths": vq_lengths
        }
        np.save(wav_path.replace('.wav', '.npy'), vq_batch)

    return vq_codes, vq_lengths

def vq_to_wav(vq_batch: Optional[dict] = None, 
              vq_path: Optional[str] = None, 
              save_wav: bool = True, 
              output_name: str = "reconstructed.wav",
              model: Optional[DAC] = None):

    if vq_batch is None and vq_path is None:
        raise ValueError("Either vq_codes or vq_path must be provided")

    if vq_path is not None:
        vq_batch = np.load(vq_path)
    
    if model is None:
        model = load_dac_model()
    
    indices = torch.from_numpy(vq_batch["vq_codes"])
    indices_lengths = torch.tensor(vq_batch["vq_lengths"])
    
    with torch.no_grad():
        fake_audio, _ = model.decode(indices, indices_lengths)
    
    if save_wav:
        sf.write(output_name, fake_audio.squeeze(1).cpu().numpy().T, 44100)
    
    return fake_audio.cpu().numpy()

if __name__ == "__main__":
    print("正在编码音频")
    vq_codes, vq_lengths = wav_to_vq("test.wav")
    print("编码完成")
    print("正在解码音频")
    fake_audio = vq_to_wav(vq_batch={"vq_codes": vq_codes, "vq_lengths": vq_lengths})
    print("解码完成")
    sf.write("reconstructed.wav", fake_audio.squeeze(1).cpu().numpy().T, 44100)
    
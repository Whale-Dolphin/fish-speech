import pytest
import torch
import numpy as np
import librosa
import soundfile as sf
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

import hydra
from omegaconf import OmegaConf
from fish_speech.models.dac.modded_dac import DAC


class TestModdedDAC:
    """Test suite for the modified DAC model encode/decode functionality."""

    @pytest.fixture
    def sample_audio(self):
        """Create a sample audio tensor for testing."""
        # Generate a 1-second sine wave at 440Hz, 44.1kHz sample rate
        sample_rate = 44100
        duration = 1.0
        frequency = 440.0
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
        return audio, sample_rate

    @pytest.fixture
    def temp_audio_file(self, sample_audio):
        """Create a temporary audio file for testing."""
        audio, sample_rate = sample_audio
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as f:
            sf.write(f.name, audio, sample_rate)
            yield f.name
        os.unlink(f.name)

    @pytest.fixture
    def model_config(self):
        """Load real model configuration from config file."""
        config_path = Path(__file__).parent.parent / \
            "fish_speech" / "configs" / "modded_dac_vq.yaml"
        return OmegaConf.load(config_path)

    def filter_state_dict_shapes(self, params, model):
        """Filter state dict to match model shapes."""
        model_state_dict = model.state_dict()
        filtered_state_dict = {
            k: v
            for k, v in params.items()
            if k in model_state_dict and v.shape == model_state_dict[k].shape
        }
        skipped_keys = set(params.keys()) - set(filtered_state_dict.keys())
        return filtered_state_dict, skipped_keys

    def test_dac_encode_decode_cycle(self, sample_audio):
        """Test the complete encode-decode cycle with real DAC model."""
        audio, sample_rate = sample_audio

        # Create a mock quantizer for testing
        mock_quantizer = MagicMock(spec=torch.nn.Module)
        mock_quantizer.return_value = MagicMock()
        mock_quantizer.return_value.codes = torch.randint(0, 1024, (1, 8, 100))
        mock_quantizer.decode.return_value = torch.randn(1, 128, 100)

        # Create a simple DAC model for testing
        test_model = DAC(
            encoder_dim=32,  # Smaller for testing
            encoder_rates=[2, 4],
            decoder_dim=128,
            decoder_rates=[4, 2],
            sample_rate=44100,
            causal=True,
            encoder_transformer_layers=[0, 0],
            decoder_transformer_layers=[0, 0],
            quantizer=mock_quantizer
        )

        # Prepare input tensor
        if len(audio.shape) == 1:
            audio = audio[None, :]
        wave_tensor = torch.from_numpy(audio).unsqueeze(1)

        # Test encode
        with torch.no_grad():
            indices, indices_lens = test_model.encode(wave_tensor)

            # Assertions for encode
            assert indices is not None
            assert indices_lens is not None
            assert isinstance(indices, torch.Tensor)
            assert isinstance(indices_lens, torch.Tensor)
            assert indices.dim() >= 2  # Should have at least batch and sequence dims

            # Test decode
            fake_audio, audio_lengths = test_model.decode(
                indices, indices_lens)

            # Assertions for decode
            assert fake_audio is not None
            assert audio_lengths is not None
            assert isinstance(fake_audio, torch.Tensor)
            assert fake_audio.dim() == 3  # [B, C, T]
            assert fake_audio.shape[0] == 1  # Batch size
            assert fake_audio.shape[1] == 1  # Channels
            assert fake_audio.shape[2] > 0   # Time dimension

    def test_audio_file_processing(self, temp_audio_file):
        """Test processing of actual audio file."""
        # Create a mock quantizer for testing
        mock_quantizer = MagicMock(spec=torch.nn.Module)
        mock_quantizer.return_value = MagicMock()
        mock_quantizer.return_value.codes = torch.randint(0, 1024, (1, 8, 50))
        mock_quantizer.decode.return_value = torch.randn(1, 128, 50)

        # Create a minimal model for testing
        test_model = DAC(
            encoder_dim=32,
            encoder_rates=[2, 4],
            decoder_dim=128,
            decoder_rates=[4, 2],
            sample_rate=44100,
            causal=True,
            encoder_transformer_layers=[0, 0],
            decoder_transformer_layers=[0, 0],
            quantizer=mock_quantizer
        )

        # Load and process audio file
        wave_np, _ = librosa.load(temp_audio_file, sr=44100, mono=True)
        if len(wave_np.shape) == 1:
            wave_np = wave_np[None, :]
        wave_tensor = torch.from_numpy(wave_np).unsqueeze(1)

        # Test the processing
        with torch.no_grad():
            indices, indices_lens = test_model.encode(wave_tensor)
            fake_audio, audio_lengths = test_model.decode(
                indices, indices_lens)

            # Test audio output format
            assert fake_audio.shape[0] == 1  # Batch size
            assert fake_audio.shape[1] == 1  # Channels
            assert fake_audio.shape[2] > 0   # Time dimension

            # Test that we can save the output (basic functionality test)
            audio_output = fake_audio.squeeze().cpu().numpy()
            if audio_output.ndim > 1:
                audio_output = audio_output[0]
            assert len(audio_output) > 0

    def test_config_file_loading(self, model_config):
        """Test that the configuration file can be loaded and contains expected keys."""
        # Test that config is loaded correctly
        assert model_config is not None
        assert "_target_" in model_config
        assert model_config._target_ == "fish_speech.models.dac.modded_dac.DAC"

        # Test required configuration keys
        required_keys = [
            "sample_rate", "encoder_dim", "encoder_rates",
            "decoder_dim", "decoder_rates", "quantizer"
        ]
        for key in required_keys:
            assert key in model_config, f"Missing required config key: {key}"

        # Test specific values
        assert model_config.sample_rate == 44100
        assert isinstance(model_config.encoder_rates, list)
        assert isinstance(model_config.decoder_rates, list)


if __name__ == "__main__":
    pytest.main([__file__])

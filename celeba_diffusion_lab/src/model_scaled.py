"""Larger RGB JiT; preserve patching, output head and flow of the small model."""
from dataclasses import asdict
from vendor_jit.config import ModelConfig, ShapeSpec
from vendor_jit.models.jit_stft import JiTSTFT


class ScaledJiT(JiTSTFT):
    def __init__(self):
        cfg = ModelConfig(in_channels=3, patch_size=(4, 4), depth=12,
                          hidden_dim=384, num_heads=6, bottleneck_dim=32,
                          output_conv_layers=4, output_conv_hidden=64)
        shape = ShapeSpec(64, 64, 64, 64, 4, 4, 16, 16, 3)
        cfg.validate()
        super().__init__(cfg, shape)
        self.in_channels = 3
        self.experiment_config = {'architecture': 'JiTSTFT scaled RGB',
                                  'model': asdict(cfg), 'shapes': asdict(shape)}

#!/usr/bin/env python3
"""
MusicGen Model Configuration
Konfiguration für AudioCraft MusicGen Training
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class MusicGenConfig:
    """MusicGen Training Configuration"""
    
    # Model Configuration
    model_name: str = "facebook/musicgen-small"
    sample_rate: int = 32000
    duration: float = 30.0
    
    # Training Configuration
    batch_size: int = 4
    learning_rate: float = 1e-4
    num_epochs: int = 5
    warmup_steps: int = 100
    
    # Data Configuration
    max_audio_length: int = 32000 * 30  # 30 seconds at 32kHz
    audio_format: str = "wav"
    
    # Paths
    data_dir: str = "data/audio"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    
    # Generation Configuration
    generation_params: dict = None
    
    def __post_init__(self):
        if self.generation_params is None:
            self.generation_params = {
                "use_sampling": True,
                "top_k": 250,
                "top_p": 0.0,
                "temperature": 1.0,
                "duration": self.duration,
                "cfg_coef": 3.0
            }

# Default configuration
default_config = MusicGenConfig()

# Small configuration for testing
small_config = MusicGenConfig(
    batch_size=2,
    num_epochs=3,
    duration=15.0,
    max_audio_length=32000 * 15
)

# Large configuration for production
large_config = MusicGenConfig(
    model_name="facebook/musicgen-medium",
    batch_size=8,
    num_epochs=10,
    duration=60.0,
    max_audio_length=32000 * 60
) 
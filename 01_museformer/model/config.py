"""
Museformer Model Configuration
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any


@dataclass
class MuseformerConfig:
    """Konfiguration für das Museformer Modell."""
    
    # Model Architecture
    vocab_size: int = 388  # MIDI vocabulary size (0-127 notes + special tokens)
    max_seq_length: int = 1024  # Maximum sequence length
    d_model: int = 512  # Model dimension
    num_heads: int = 8  # Number of attention heads
    num_layers: int = 12  # Number of transformer layers
    d_ff: int = 2048  # Feed-forward dimension
    dropout: float = 0.1  # Dropout rate
    
    # Bar-Level Attention
    use_bar_attention: bool = True  # Enable bar-level attention
    bars_per_segment: int = 4  # Number of bars per attention segment
    beats_per_bar: int = 4  # Time signature (4/4)
    ticks_per_beat: int = 480  # MIDI ticks per beat
    
    # Multi-Track Support
    num_tracks: int = 4  # Number of MIDI tracks
    track_embedding_dim: int = 64  # Track embedding dimension
    
    # Positional Encoding
    max_position_embeddings: int = 2048
    use_relative_position: bool = True
    
    # Special Tokens
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    unk_token_id: int = 3
    bar_token_id: int = 4
    track_token_id: int = 5
    
    # Training Configuration
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    
    # Generation Parameters
    max_new_tokens: int = 512
    do_sample: bool = True
    temperature: float = 1.0
    top_k: int = 50
    top_p: float = 0.95
    repetition_penalty: float = 1.1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            field.name: getattr(self, field.name)
            for field in self.__dataclass_fields__.values()
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MuseformerConfig':
        """Create config from dictionary."""
        return cls(**config_dict)


# Ultra-small config für 6GB GPU
ULTRA_SMALL_CONFIG = MuseformerConfig(
    vocab_size=283,  # Wird automatisch angepasst
    d_model=256,  # Noch kleiner
    num_layers=4,  # Weniger Layer
    num_heads=4,  # Weniger Attention Heads
    d_ff=512,  # Kleiner FFN
    max_seq_length=256,  # Kürzere Sequenzen
    max_position_embeddings=512,  # MUSS größer sein als max_seq_length!
    num_tracks=4,  # Weniger Tracks
    bars_per_segment=4,  # Weniger Bars
    use_relative_position=False,  # Verwende absolute Positional Encoding
    dropout=0.1,
    pad_token_id=0,
    bos_token_id=1,
    eos_token_id=2
)

# Predefined configurations for different use cases
CONFIGS = {
    'ultra_small': ULTRA_SMALL_CONFIG,
    "small": MuseformerConfig(
        d_model=256,
        num_heads=4,
        num_layers=6,
        d_ff=1024,
        max_seq_length=512
    ),
    
    "base": MuseformerConfig(
        d_model=512,
        num_heads=8,
        num_layers=12,
        d_ff=2048,
        max_seq_length=1024
    ),
    
    "large": MuseformerConfig(
        d_model=768,
        num_heads=12,
        num_layers=18,
        d_ff=3072,
        max_seq_length=1536
    ),
    
    "pop_style": MuseformerConfig(
        # Optimized for pop music generation
        d_model=512,
        num_heads=8,
        num_layers=12,
        bars_per_segment=8,  # Longer attention for pop structure
        num_tracks=6,  # More tracks for rich arrangements
        temperature=0.8,  # Less randomness for coherent pop
        top_p=0.9
    )
}


def get_config(config_name: str = "base") -> MuseformerConfig:
    """Get predefined configuration."""
    if config_name not in CONFIGS:
        raise ValueError(f"Unknown config: {config_name}. Available: {list(CONFIGS.keys())}")
    return CONFIGS[config_name] 
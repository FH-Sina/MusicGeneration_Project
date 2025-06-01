#!/usr/bin/env python3
"""
MusicGen Model Package
Audio Music Generation Components
"""

from .config import MusicGenConfig, default_config
from .musicgen import MusicGenWrapper, create_musicgen_model
from .attention import (
    MultiHeadAttention, 
    CrossAttention, 
    TemporalAttention, 
    AudioAttentionBlock,
    create_causal_mask,
    create_padding_mask
)
from .embedding import (
    AudioTokenEmbedding,
    PositionalEncoding,
    LearnablePositionalEncoding,
    ConditioningEmbedding,
    MultiCodebookEmbedding,
    AudioEmbeddingLayer,
    create_embedding_layer
)

__all__ = [
    'MusicGenConfig',
    'default_config',
    'MusicGenWrapper',
    'create_musicgen_model',
    'MultiHeadAttention',
    'CrossAttention',
    'TemporalAttention',
    'AudioAttentionBlock',
    'create_causal_mask',
    'create_padding_mask',
    'AudioTokenEmbedding',
    'PositionalEncoding',
    'LearnablePositionalEncoding',
    'ConditioningEmbedding',
    'MultiCodebookEmbedding',
    'AudioEmbeddingLayer',
    'create_embedding_layer'
] 
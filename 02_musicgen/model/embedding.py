#!/usr/bin/env python3
"""
Embedding Layers für MusicGen
Audio-Token Embeddings und Positional Encodings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any

class AudioTokenEmbedding(nn.Module):
    """
    Audio Token Embedding für MusicGen
    Ähnlich der MIDI Token Embeddings aber für Audio-Codes
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None,
        max_norm: Optional[float] = None
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.padding_idx = padding_idx
        
        self.embedding = nn.Embedding(
            vocab_size, 
            d_model, 
            padding_idx=padding_idx,
            max_norm=max_norm
        )
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialisiert die Embedding-Gewichte"""
        nn.init.normal_(self.embedding.weight, mean=0, std=self.d_model ** -0.5)
        if self.padding_idx is not None:
            nn.init.constant_(self.embedding.weight[self.padding_idx], 0)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            tokens: Audio token IDs [batch_size, seq_len]
            
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        return self.embedding(tokens) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding für Audio-Sequenzen
    Erweitert für längere Audio-Sequenzen
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, seq_len, d_model]
        Returns:
            x with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :].to(x.device)
        return self.dropout(x)

class LearnablePositionalEncoding(nn.Module):
    """
    Lernbare Positional Encodings
    Alternative zu sinusoidalen Encodings
    """
    
    def __init__(
        self,
        d_model: int,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Parameter(torch.randn(1, max_seq_len, d_model))
        
        # Initialize
        nn.init.normal_(self.pe, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)

class ConditioningEmbedding(nn.Module):
    """
    Embedding für Text-Konditionierung
    Verarbeitet Text-Prompts für MusicGen
    """
    
    def __init__(
        self,
        text_vocab_size: int,
        d_model: int,
        max_text_len: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        self.text_embedding = nn.Embedding(text_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_text_len, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Text projection
        self.text_projection = nn.Linear(d_model, d_model)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: [batch_size, text_len]
        Returns:
            text_embeddings: [batch_size, text_len, d_model]
        """
        # Text embeddings
        text_emb = self.text_embedding(text_tokens)
        text_emb = self.positional_encoding(text_emb)
        text_emb = self.layer_norm(text_emb)
        
        # Project to model dimension
        text_emb = self.text_projection(text_emb)
        
        return text_emb

class MultiCodebookEmbedding(nn.Module):
    """
    Multi-Codebook Embedding für MusicGen
    Verarbeitet mehrere Audio-Codebooks parallel
    """
    
    def __init__(
        self,
        num_codebooks: int,
        vocab_size: int,
        d_model: int,
        padding_idx: Optional[int] = None
    ):
        super().__init__()
        self.num_codebooks = num_codebooks
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Separate embedding for each codebook
        self.embeddings = nn.ModuleList([
            nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
            for _ in range(num_codebooks)
        ])
        
        # Combination layer
        self.combination = nn.Linear(num_codebooks * d_model, d_model)
        
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialisiert alle Embedding-Gewichte"""
        for embedding in self.embeddings:
            nn.init.normal_(embedding.weight, mean=0, std=self.d_model ** -0.5)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            tokens: [batch_size, seq_len, num_codebooks]
        Returns:
            embeddings: [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, num_codebooks = tokens.shape
        assert num_codebooks == self.num_codebooks
        
        # Get embeddings for each codebook
        codebook_embeddings = []
        for i, embedding in enumerate(self.embeddings):
            emb = embedding(tokens[:, :, i])  # [batch_size, seq_len, d_model]
            codebook_embeddings.append(emb)
        
        # Concatenate and combine
        combined = torch.cat(codebook_embeddings, dim=-1)  # [batch_size, seq_len, num_codebooks * d_model]
        output = self.combination(combined)  # [batch_size, seq_len, d_model]
        
        return output * math.sqrt(self.d_model)

class AudioEmbeddingLayer(nn.Module):
    """
    Vollständige Embedding-Schicht für MusicGen
    Kombiniert Token-, Positional- und Conditioning-Embeddings
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_seq_len: int = 8192,
        num_codebooks: int = 4,
        text_vocab_size: Optional[int] = None,
        dropout: float = 0.1,
        use_learnable_pe: bool = False
    ):
        super().__init__()
        
        # Audio token embeddings
        if num_codebooks > 1:
            self.token_embedding = MultiCodebookEmbedding(
                num_codebooks, vocab_size, d_model
            )
        else:
            self.token_embedding = AudioTokenEmbedding(vocab_size, d_model)
        
        # Positional encoding
        if use_learnable_pe:
            self.positional_encoding = LearnablePositionalEncoding(
                d_model, max_seq_len, dropout
            )
        else:
            self.positional_encoding = PositionalEncoding(
                d_model, max_seq_len, dropout
            )
        
        # Text conditioning (optional)
        self.text_conditioning = None
        if text_vocab_size is not None:
            self.text_conditioning = ConditioningEmbedding(
                text_vocab_size, d_model, dropout=dropout
            )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_tokens: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            audio_tokens: Audio token IDs
            text_tokens: Text token IDs (optional)
            
        Returns:
            Dict with 'audio_embeddings' and optionally 'text_embeddings'
        """
        # Audio embeddings
        audio_emb = self.token_embedding(audio_tokens)
        audio_emb = self.positional_encoding(audio_emb)
        audio_emb = self.layer_norm(audio_emb)
        audio_emb = self.dropout(audio_emb)
        
        result = {'audio_embeddings': audio_emb}
        
        # Text embeddings (if provided)
        if text_tokens is not None and self.text_conditioning is not None:
            text_emb = self.text_conditioning(text_tokens)
            result['text_embeddings'] = text_emb
        
        return result

def create_embedding_layer(
    vocab_size: int,
    d_model: int,
    **kwargs
) -> AudioEmbeddingLayer:
    """Factory function für Audio Embedding Layer"""
    return AudioEmbeddingLayer(vocab_size, d_model, **kwargs) 
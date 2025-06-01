#!/usr/bin/env python3
"""
Attention Mechanisms für MusicGen
Erweiterte Attention-Komponenten für Audio-Generierung
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention für MusicGen
    Ähnlich der Museformer Attention aber für Audio-Tokens
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        batch_size, seq_len = query.size(0), query.size(1)
        
        # Linear projections
        Q = self.w_q(query).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention
        attention_output = self.scaled_dot_product_attention(Q, K, V, mask)
        
        # Concatenate heads
        attention_output = attention_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.w_o(attention_output)
    
    def scaled_dot_product_attention(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        # Softmax
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        return torch.matmul(attention_weights, V)

class CrossAttention(nn.Module):
    """
    Cross-Attention für Text-zu-Audio Konditionierung
    Spezifisch für MusicGen's Text-Prompts
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        audio_tokens: torch.Tensor,
        text_embeddings: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Cross-attention: audio queries, text keys/values
        attended = self.attention(
            query=audio_tokens,
            key=text_embeddings,
            value=text_embeddings,
            mask=mask
        )
        
        # Residual connection and normalization
        audio_tokens = self.norm1(audio_tokens + attended)
        
        # Feed-forward network
        ffn_output = self.ffn(audio_tokens)
        audio_tokens = self.norm2(audio_tokens + ffn_output)
        
        return audio_tokens

class TemporalAttention(nn.Module):
    """
    Temporal Attention für Audio-Sequenzen
    Berücksichtigt zeitliche Abhängigkeiten in Audio
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int = 8192,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.positional_encoding = self._create_positional_encoding(max_seq_len, d_model)
        
    def _create_positional_encoding(self, max_seq_len: int, d_model: int) -> torch.Tensor:
        """Erstellt Positional Encoding für Audio-Sequenzen"""
        pe = torch.zeros(max_seq_len, d_model)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           -(math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)  # Add batch dimension
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        seq_len = x.size(1)
        
        # Add positional encoding
        if seq_len <= self.positional_encoding.size(1):
            pos_encoding = self.positional_encoding[:, :seq_len, :].to(x.device)
            x = x + pos_encoding
        
        # Self-attention
        return self.attention(x, x, x, mask)

class AudioAttentionBlock(nn.Module):
    """
    Vollständiger Attention Block für Audio-Generierung
    Kombiniert Self-Attention, Cross-Attention und FFN
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
        use_cross_attention: bool = True
    ):
        super().__init__()
        
        # Self-attention
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        
        # Cross-attention (optional)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention = CrossAttention(d_model, num_heads, dropout)
            self.norm2 = nn.LayerNorm(d_model)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        self_attn_mask: Optional[torch.Tensor] = None,
        cross_attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        self_attended = self.self_attention(x, x, x, self_attn_mask)
        x = self.norm1(x + self.dropout(self_attended))
        
        # Cross-attention (if enabled and text provided)
        if self.use_cross_attention and text_embeddings is not None:
            cross_attended = self.cross_attention(x, text_embeddings, cross_attn_mask)
            x = self.norm2(x + cross_attended)
        
        # Feed-forward
        ffn_output = self.ffn(x)
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x

def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Erstellt eine Causal Mask für autoregressive Generierung"""
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.unsqueeze(0).unsqueeze(0)  # Add batch and head dimensions

def create_padding_mask(lengths: torch.Tensor, max_len: int) -> torch.Tensor:
    """Erstellt eine Padding Mask für variable Sequenzlängen"""
    batch_size = lengths.size(0)
    mask = torch.arange(max_len, device=lengths.device).expand(
        batch_size, max_len
    ) < lengths.unsqueeze(1)
    return mask.unsqueeze(1).unsqueeze(1)  # Add head and query dimensions 
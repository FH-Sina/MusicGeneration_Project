"""
Bar-Level Attention Mechanisms für Museformer
Implementiert spezielle Aufmerksamkeitsmuster für musikalische Strukturen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class BarLevelAttention(nn.Module):
    """
    Bar-Level Attention für strukturierte Musikgenerierung.
    Kombiniert lokale (innerhalb des Takts) und globale (zwischen Takten) Aufmerksamkeit.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 bars_per_segment: int = 4,
                 ticks_per_bar: int = 1920,
                 dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            bars_per_segment: Number of bars per attention segment
            ticks_per_bar: MIDI ticks per bar (4 beats * 480 ticks)
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.bars_per_segment = bars_per_segment
        self.ticks_per_bar = ticks_per_bar
        
        # Projection layers
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Bar position encoding
        self.bar_pos_encoding = nn.Embedding(bars_per_segment, d_model)
        
        # Local vs global attention weights
        self.local_global_gate = nn.Parameter(torch.randn(num_heads))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_head)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                bar_positions: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            bar_positions: Bar position indices [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        # Local attention (within bars)
        local_attn = self._compute_local_attention(q, k, v, bar_positions, attention_mask)
        
        # Global attention (between bars)
        global_attn = self._compute_global_attention(q, k, v, bar_positions, attention_mask)
        
        # Combine local and global attention
        gate = torch.sigmoid(self.local_global_gate).view(1, self.num_heads, 1, 1)
        combined_attn = gate * local_attn + (1 - gate) * global_attn
        
        # Reshape back
        output = combined_attn.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.out_proj(output)
    
    def _compute_local_attention(self, 
                               q: torch.Tensor,
                               k: torch.Tensor, 
                               v: torch.Tensor,
                               bar_positions: Optional[torch.Tensor],
                               attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Berechnet lokale Aufmerksamkeit innerhalb von Takten."""
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if bar_positions is not None:
            # Create mask for same-bar positions
            bar_mask = (bar_positions.unsqueeze(-1) == bar_positions.unsqueeze(-2))
            bar_mask = bar_mask.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
            
            # Only attend within the same bar
            scores = scores.masked_fill(~bar_mask, float('-inf'))
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)
    
    def _compute_global_attention(self,
                                q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                bar_positions: Optional[torch.Tensor],
                                attention_mask: Optional[torch.Tensor]) -> torch.Tensor:
        """Berechnet globale Aufmerksamkeit zwischen Takten."""
        # Standard scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if bar_positions is not None:
            # Add bar positional bias
            batch_size, seq_len = bar_positions.shape
            bar_pos_emb = self.bar_pos_encoding(bar_positions % self.bars_per_segment)
            
            # Compute bar-level bias
            bar_bias = torch.matmul(
                bar_pos_emb.unsqueeze(-2),
                bar_pos_emb.unsqueeze(-1)
            ).squeeze(-1)
            
            scores = scores + bar_bias.unsqueeze(1)
        
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.unsqueeze(1).unsqueeze(1), float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        return torch.matmul(attn_weights, v)


class RelativePositionalEncoding(nn.Module):
    """
    Relative Positional Encoding für Musiksequenzen.
    Berücksichtigt musikalische Zeitabstände.
    """
    
    def __init__(self, d_model: int, max_relative_distance: int = 128):
        super().__init__()
        self.d_model = d_model
        self.max_relative_distance = max_relative_distance
        
        # Relative position embeddings
        self.relative_pos_emb = nn.Embedding(
            2 * max_relative_distance + 1, 
            d_model
        )
        
    def forward(self, seq_len: int) -> torch.Tensor:
        """
        Erstellt relative Positionsembeddings.
        
        Args:
            seq_len: Sequence length
            
        Returns:
            Relative position embeddings [seq_len, seq_len, d_model]
        """
        # Create relative position matrix
        positions = torch.arange(seq_len, device=self.relative_pos_emb.weight.device)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        
        # Clip to maximum distance
        relative_positions = torch.clamp(
            relative_positions,
            -self.max_relative_distance,
            self.max_relative_distance
        )
        
        # Shift to positive indices
        relative_positions = relative_positions + self.max_relative_distance
        
        return self.relative_pos_emb(relative_positions)


class MultiScaleAttention(nn.Module):
    """
    Multi-Scale Attention für verschiedene musikalische Zeitebenen.
    Kombiniert Beat-, Bar-, und Phrase-Level Attention.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 scales: list = [1, 4, 16],  # beat, bar, phrase
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.scales = scales
        self.num_scales = len(scales)
        
        # Separate attention for each scale
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(
                d_model, 
                num_heads // self.num_scales,
                dropout=dropout,
                batch_first=True
            )
            for _ in scales
        ])
        
        # Scale fusion
        self.scale_fusion = nn.Linear(d_model * self.num_scales, d_model)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        scale_outputs = []
        
        for scale, scale_attn in zip(self.scales, self.scale_attentions):
            # Downsample for larger scales
            if scale > 1:
                # Average pooling
                downsampled_len = seq_len // scale
                downsampled = F.avg_pool1d(
                    hidden_states.transpose(1, 2),
                    kernel_size=scale,
                    stride=scale
                ).transpose(1, 2)
                
                # Apply attention
                scale_out, _ = scale_attn(downsampled, downsampled, downsampled)
                
                # Upsample back
                scale_out = F.interpolate(
                    scale_out.transpose(1, 2),
                    size=seq_len,
                    mode='linear',
                    align_corners=False
                ).transpose(1, 2)
            else:
                # Original scale
                scale_out, _ = scale_attn(hidden_states, hidden_states, hidden_states)
            
            scale_outputs.append(scale_out)
        
        # Combine all scales
        combined = torch.cat(scale_outputs, dim=-1)
        output = self.scale_fusion(combined)
        
        return output


class StructuralAttention(nn.Module):
    """
    Structural Attention für musikalische Formstrukturen.
    Erkennt und nutzt repetitive Muster in der Musik.
    """
    
    def __init__(self, 
                 d_model: int,
                 num_heads: int,
                 structure_length: int = 32,  # Length of structural units
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.structure_length = structure_length
        
        # Standard multi-head attention
        self.attention = nn.MultiheadAttention(
            d_model, num_heads, dropout=dropout, batch_first=True
        )
        
        # Structural pattern detection
        self.pattern_detector = nn.Conv1d(
            d_model, d_model, kernel_size=structure_length, 
            stride=structure_length, padding=0
        )
        
        # Pattern fusion
        self.pattern_fusion = nn.Linear(d_model * 2, d_model)
        
    def forward(self, 
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Attention mask [batch_size, seq_len]
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = hidden_states.shape
        
        # Standard attention
        attn_out, _ = self.attention(hidden_states, hidden_states, hidden_states)
        
        # Detect structural patterns
        patterns = self.pattern_detector(hidden_states.transpose(1, 2))
        patterns = patterns.transpose(1, 2)
        
        # Repeat patterns to match sequence length
        num_patterns = patterns.shape[1]
        pattern_repeated = patterns.repeat_interleave(
            self.structure_length, dim=1
        )[:, :seq_len, :]
        
        # Pad if necessary
        if pattern_repeated.shape[1] < seq_len:
            padding = torch.zeros(
                batch_size, 
                seq_len - pattern_repeated.shape[1], 
                d_model,
                device=hidden_states.device
            )
            pattern_repeated = torch.cat([pattern_repeated, padding], dim=1)
        
        # Combine attention output with structural patterns
        combined = torch.cat([attn_out, pattern_repeated], dim=-1)
        output = self.pattern_fusion(combined)
        
        return output 
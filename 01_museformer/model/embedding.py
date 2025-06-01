"""
Embedding Modules für Museformer
MIDI Token Embeddings und Track Embeddings.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MIDIEmbedding(nn.Module):
    """
    MIDI Token Embedding mit speziellen Eigenschaften für Musikdaten.
    """
    
    def __init__(self, 
                 vocab_size: int,
                 d_model: int,
                 pad_token_id: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        
        # Standard token embedding
        self.token_embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_token_id)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Scale embeddings by sqrt(d_model) wie in Transformer
        self.scale = math.sqrt(d_model)
        
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            
        Returns:
            Token embeddings [batch_size, seq_len, d_model]
        """
        embeddings = self.token_embedding(input_ids) * self.scale
        return self.dropout(embeddings)


class TrackEmbedding(nn.Module):
    """
    Track-spezifische Embeddings für Multi-Track MIDI.
    """
    
    def __init__(self, 
                 num_tracks: int,
                 embedding_dim: int,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_tracks = num_tracks
        self.embedding_dim = embedding_dim
        
        # Track embedding
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        
        # Optional: Instrument-specific embeddings
        # General MIDI hat 128 Instrumente
        self.instrument_embedding = nn.Embedding(128, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                track_ids: torch.Tensor,
                instrument_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            track_ids: Track IDs [batch_size, seq_len]
            instrument_ids: Optional instrument IDs [batch_size, seq_len]
            
        Returns:
            Track embeddings [batch_size, seq_len, embedding_dim]
        """
        # Track embeddings
        track_emb = self.track_embedding(track_ids)
        
        # Füge Instrument-Embeddings hinzu falls verfügbar
        if instrument_ids is not None:
            instrument_emb = self.instrument_embedding(instrument_ids)
            track_emb = track_emb + instrument_emb
        
        return self.dropout(track_emb)


class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding für Musiksequenzen.
    """
    
    def __init__(self, 
                 d_model: int, 
                 max_length: int = 5000,
                 dropout: float = 0.1):
        super().__init__()
        
        self.dropout = nn.Dropout(dropout)
        
        # Erstelle Positional Encoding Matrix
        pe = torch.zeros(max_length, d_model)
        position = torch.arange(0, max_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * 
            (-math.log(10000.0) / d_model)
        )
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        # Registriere als Buffer (wird nicht trainiert)
        self.register_buffer('pe', pe)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor [seq_len, batch_size, d_model]
            
        Returns:
            Tensor mit Positional Encoding [seq_len, batch_size, d_model]
        """
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class MusicalPositionalEncoding(nn.Module):
    """
    Musikalisch-informierte Positional Encoding.
    Berücksichtigt Beat-, Bar- und Phrase-Strukturen.
    """
    
    def __init__(self,
                 d_model: int,
                 max_length: int = 5000,
                 beats_per_bar: int = 4,
                 ticks_per_beat: int = 480,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.beats_per_bar = beats_per_bar
        self.ticks_per_beat = ticks_per_beat
        self.ticks_per_bar = beats_per_bar * ticks_per_beat
        
        # Standard positional encoding
        self.pos_encoding = PositionalEncoding(d_model // 3, max_length, dropout=0.0)
        
        # Beat-level encoding
        self.beat_encoding = nn.Embedding(beats_per_bar, d_model // 3)
        
        # Bar-level encoding (modulo 16 für längere Sequenzen)
        self.bar_encoding = nn.Embedding(16, d_model // 3)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                positions: torch.Tensor,
                beat_positions: Optional[torch.Tensor] = None,
                bar_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            positions: Absolute positions [batch_size, seq_len]
            beat_positions: Beat positions within bar [batch_size, seq_len]
            bar_positions: Bar positions [batch_size, seq_len]
            
        Returns:
            Musical positional encoding [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = positions.shape
        
        # Standard positional encoding
        pos_emb = self.pos_encoding.pe[positions].squeeze(1)  # [batch_size, seq_len, d_model//3]
        
        # Beat-level encoding
        if beat_positions is None:
            beat_positions = (positions // self.ticks_per_beat) % self.beats_per_bar
        beat_emb = self.beat_encoding(beat_positions)
        
        # Bar-level encoding
        if bar_positions is None:
            bar_positions = (positions // self.ticks_per_bar) % 16
        bar_emb = self.bar_encoding(bar_positions)
        
        # Kombiniere alle Encodings
        combined = torch.cat([pos_emb, beat_emb, bar_emb], dim=-1)
        
        return self.dropout(combined)


class VelocityEmbedding(nn.Module):
    """
    Velocity-spezifische Embeddings für expressive MIDI-Performance.
    """
    
    def __init__(self, 
                 num_velocity_bins: int = 32,
                 embedding_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_velocity_bins = num_velocity_bins
        self.embedding_dim = embedding_dim
        
        # Velocity embedding
        self.velocity_embedding = nn.Embedding(num_velocity_bins, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, velocity_bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            velocity_bins: Quantisierte Velocity-Werte [batch_size, seq_len]
            
        Returns:
            Velocity embeddings [batch_size, seq_len, embedding_dim]
        """
        velocity_emb = self.velocity_embedding(velocity_bins)
        return self.dropout(velocity_emb)


class DurationEmbedding(nn.Module):
    """
    Duration-spezifische Embeddings für Note-Längen.
    """
    
    def __init__(self, 
                 num_duration_bins: int = 64,
                 embedding_dim: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        self.num_duration_bins = num_duration_bins
        self.embedding_dim = embedding_dim
        
        # Duration embedding
        self.duration_embedding = nn.Embedding(num_duration_bins, embedding_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, duration_bins: torch.Tensor) -> torch.Tensor:
        """
        Args:
            duration_bins: Quantisierte Duration-Werte [batch_size, seq_len]
            
        Returns:
            Duration embeddings [batch_size, seq_len, embedding_dim]
        """
        duration_emb = self.duration_embedding(duration_bins)
        return self.dropout(duration_emb)


class CompoundMIDIEmbedding(nn.Module):
    """
    Kombiniertes MIDI Embedding das mehrere Aspekte berücksichtigt:
    - Token (Note, Velocity, Duration, etc.)
    - Track Information
    - Positional Information
    - Musical Structure
    """
    
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 num_tracks: int = 16,
                 max_length: int = 2048,
                 num_velocity_bins: int = 32,
                 num_duration_bins: int = 64,
                 pad_token_id: int = 0,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Hauptkomponenten
        component_dim = d_model // 4
        
        # Token embedding
        self.token_embedding = MIDIEmbedding(
            vocab_size, component_dim, pad_token_id, dropout
        )
        
        # Track embedding
        self.track_embedding = TrackEmbedding(
            num_tracks, component_dim, dropout
        )
        
        # Musical positional encoding
        self.pos_encoding = MusicalPositionalEncoding(
            component_dim, max_length, dropout=dropout
        )
        
        # Velocity embedding
        self.velocity_embedding = VelocityEmbedding(
            num_velocity_bins, component_dim, dropout
        )
        
        # Projektion auf finale Dimension
        self.projection = nn.Linear(component_dim * 4, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self,
                input_ids: torch.Tensor,
                track_ids: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None,
                velocity_bins: Optional[torch.Tensor] = None,
                beat_positions: Optional[torch.Tensor] = None,
                bar_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            track_ids: Track IDs [batch_size, seq_len]
            positions: Absolute positions [batch_size, seq_len]
            velocity_bins: Velocity bins [batch_size, seq_len]
            beat_positions: Beat positions [batch_size, seq_len]
            bar_positions: Bar positions [batch_size, seq_len]
            
        Returns:
            Combined embeddings [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Token embeddings
        token_emb = self.token_embedding(input_ids)
        
        # Track embeddings
        if track_ids is None:
            track_ids = torch.zeros_like(input_ids)
        track_emb = self.track_embedding(track_ids)
        
        # Positional embeddings
        if positions is None:
            positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        pos_emb = self.pos_encoding(positions, beat_positions, bar_positions)
        
        # Velocity embeddings
        if velocity_bins is None:
            velocity_bins = torch.zeros_like(input_ids)
        vel_emb = self.velocity_embedding(velocity_bins)
        
        # Kombiniere alle Embeddings
        combined = torch.cat([token_emb, track_emb, pos_emb, vel_emb], dim=-1)
        
        # Projiziere auf finale Dimension
        output = self.projection(combined)
        
        return self.dropout(output) 
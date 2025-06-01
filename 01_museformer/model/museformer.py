"""
Museformer: Transformer-basiertes Modell für symbolische Musikgenerierung
Hauptmodell-Implementierung mit Bar-Level Attention und Multi-Track Support.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any, Tuple, List
from dataclasses import dataclass

from .config import MuseformerConfig
from .attention import BarLevelAttention, RelativePositionalEncoding, MultiScaleAttention
from .embedding import MIDIEmbedding, TrackEmbedding


@dataclass
class MuseformerOutput:
    """Output des Museformer Modells."""
    last_hidden_state: torch.Tensor
    logits: torch.Tensor
    hidden_states: Optional[Tuple[torch.Tensor]] = None
    attentions: Optional[Tuple[torch.Tensor]] = None
    loss: Optional[torch.Tensor] = None


class MuseformerBlock(nn.Module):
    """
    Ein Transformer-Block des Museformer Modells.
    Kombiniert Bar-Level Attention mit Standard Feed-Forward Netzwerk.
    """
    
    def __init__(self, config: MuseformerConfig):
        super().__init__()
        self.config = config
        
        # Attention layer
        if config.use_bar_attention:
            self.attention = BarLevelAttention(
                d_model=config.d_model,
                num_heads=config.num_heads,
                bars_per_segment=config.bars_per_segment,
                dropout=config.dropout
            )
        else:
            self.attention = nn.MultiheadAttention(
                config.d_model,
                config.num_heads,
                dropout=config.dropout,
                batch_first=True
            )
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(config.d_model, config.d_ff),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_ff, config.d_model),
            nn.Dropout(config.dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.d_model)
        self.norm2 = nn.LayerNorm(config.d_model)
        
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
        # Self-attention with residual connection
        normed_hidden = self.norm1(hidden_states)
        
        if self.config.use_bar_attention:
            attn_output = self.attention(
                normed_hidden, 
                bar_positions=bar_positions,
                attention_mask=attention_mask
            )
        else:
            attn_output, _ = self.attention(
                normed_hidden, 
                normed_hidden, 
                normed_hidden,
                key_padding_mask=~attention_mask if attention_mask is not None else None
            )
        
        hidden_states = hidden_states + attn_output
        
        # Feed-forward with residual connection
        normed_hidden = self.norm2(hidden_states)
        ff_output = self.feed_forward(normed_hidden)
        hidden_states = hidden_states + ff_output
        
        return hidden_states


class MuseformerModel(nn.Module):
    """
    Hauptmodell des Museformer für symbolische Musikgenerierung.
    """
    
    def __init__(self, config: MuseformerConfig):
        super().__init__()
        self.config = config
        
        # Token embeddings
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        
        # Track embeddings für Multi-Track Support
        self.track_embedding = nn.Embedding(config.num_tracks, config.track_embedding_dim)
        
        # Track projection layer (if needed)
        if config.track_embedding_dim != config.d_model:
            self.track_projection = nn.Linear(config.track_embedding_dim, config.d_model)
        else:
            self.track_projection = None
        
        # Positional encoding
        if config.use_relative_position:
            self.pos_encoding = RelativePositionalEncoding(config.d_model)
        else:
            self.pos_encoding = nn.Embedding(config.max_position_embeddings, config.d_model)
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            MuseformerBlock(config) for _ in range(config.num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(config.d_model)
        
        # Output projection
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Gewichtsinitialisierung."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                track_ids: Optional[torch.Tensor] = None,
                bar_positions: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                output_hidden_states: bool = False,
                output_attentions: bool = False) -> MuseformerOutput:
        """
        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            track_ids: Track IDs [batch_size, seq_len]
            bar_positions: Bar position indices [batch_size, seq_len]
            labels: Target token IDs für Training [batch_size, seq_len]
            output_hidden_states: Ob alle Hidden States zurückgegeben werden sollen
            output_attentions: Ob Attention Weights zurückgegeben werden sollen
            
        Returns:
            MuseformerOutput mit Logits und optionalen Zusatzinformationen
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Track embeddings hinzufügen
        if track_ids is not None:
            # Ensure track_ids don't exceed num_tracks
            track_ids = torch.clamp(track_ids, 0, self.config.num_tracks - 1)
            track_emb = self.track_embedding(track_ids)
            if self.track_projection is not None:
                track_emb = self.track_projection(track_emb)
            hidden_states = hidden_states + track_emb
        
        # Positional encoding
        if self.config.use_relative_position:
            # Relative positional encoding wird in den Attention-Layern verwendet
            pass
        else:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
            # Ensure position_ids don't exceed max_position_embeddings
            position_ids = torch.clamp(position_ids, 0, self.config.max_position_embeddings - 1)
            pos_emb = self.pos_encoding(position_ids)
            hidden_states = hidden_states + pos_emb
        
        # Dropout
        hidden_states = self.dropout(hidden_states)
        
        # Durchlaufe alle Transformer-Blöcke
        all_hidden_states = []
        all_attentions = []
        
        for block in self.blocks:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            hidden_states = block(
                hidden_states,
                bar_positions=bar_positions,
                attention_mask=attention_mask
            )
        
        # Final layer norm
        hidden_states = self.final_norm(hidden_states)
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Output projection
        logits = self.lm_head(hidden_states)
        
        # Berechne Loss falls Labels gegeben sind
        loss = None
        if labels is not None:
            # Shift labels for causal language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Flatten für CrossEntropy
            loss_fct = nn.CrossEntropyLoss(
                ignore_index=self.config.pad_token_id,
                label_smoothing=self.config.label_smoothing
            )
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        return MuseformerOutput(
            last_hidden_state=hidden_states,
            logits=logits,
            hidden_states=tuple(all_hidden_states) if output_hidden_states else None,
            attentions=tuple(all_attentions) if output_attentions else None,
            loss=loss
        )
    
    def generate(self,
                 input_ids: torch.Tensor,
                 max_new_tokens: int = 512,
                 do_sample: bool = True,
                 temperature: float = 1.0,
                 top_k: int = 50,
                 top_p: float = 0.95,
                 repetition_penalty: float = 1.1,
                 pad_token_id: Optional[int] = None,
                 eos_token_id: Optional[int] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 track_ids: Optional[torch.Tensor] = None,
                 bar_positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generiert neue Token-Sequenzen.
        
        Args:
            input_ids: Eingabe-Token [batch_size, seq_len]
            max_new_tokens: Maximale Anzahl neuer Token
            do_sample: Ob Sampling verwendet werden soll
            temperature: Sampling-Temperatur
            top_k: Top-K Sampling
            top_p: Nucleus Sampling
            repetition_penalty: Penalty für Wiederholungen
            pad_token_id: Padding Token ID
            eos_token_id: End-of-Sequence Token ID
            attention_mask: Attention Mask
            track_ids: Track IDs
            bar_positions: Bar Positions
            
        Returns:
            Generierte Token-Sequenz [batch_size, seq_len + max_new_tokens]
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        # Kopiere input für Generierung
        generated = input_ids.clone()
        
        # Erweitere Hilfstensoren falls notwendig
        if attention_mask is not None:
            attention_mask = attention_mask.clone()
        if track_ids is not None:
            track_ids = track_ids.clone()
        if bar_positions is not None:
            bar_positions = bar_positions.clone()
        
        for _ in range(max_new_tokens):
            # Forward pass
            outputs = self.forward(
                input_ids=generated,
                attention_mask=attention_mask,
                track_ids=track_ids,
                bar_positions=bar_positions
            )
            
            # Nächstes Token vorhersagen
            next_token_logits = outputs.logits[:, -1, :]
            
            # Repetition penalty anwenden
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in set(generated[i].tolist()):
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty
            
            # Sampling oder Greedy
            if do_sample:
                # Temperature scaling
                next_token_logits = next_token_logits / temperature
                
                # Top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Entferne Token mit kumulativer Wahrscheinlichkeit über threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(
                        1, sorted_indices, sorted_indices_to_remove
                    )
                    next_token_logits = next_token_logits.masked_fill(indices_to_remove, float('-inf'))
                
                # Sample from distribution
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Füge nächstes Token hinzu
            generated = torch.cat([generated, next_token], dim=-1)
            
            # Erweitere Hilfstensoren
            if attention_mask is not None:
                attention_mask = torch.cat([
                    attention_mask, 
                    torch.ones(batch_size, 1, device=device)
                ], dim=-1)
            
            if track_ids is not None:
                # Behalte letzten Track bei
                last_track = track_ids[:, -1:]
                track_ids = torch.cat([track_ids, last_track], dim=-1)
            
            if bar_positions is not None:
                # Aktualisiere Bar-Position basierend auf Token-Typ
                # Vereinfachte Logik - kann erweitert werden
                last_bar = bar_positions[:, -1:]
                bar_positions = torch.cat([bar_positions, last_bar], dim=-1)
            
            # Stoppe bei EOS token
            if (next_token == eos_token_id).all():
                break
        
        return generated


class MuseformerForCausalLM(nn.Module):
    """
    Museformer-Wrapper für Causal Language Modeling.
    Kompatibel mit HuggingFace Transformers Interface.
    """
    
    def __init__(self, config: MuseformerConfig):
        super().__init__()
        self.config = config
        self.model = MuseformerModel(config)
        
    def forward(self, **kwargs):
        """Forward pass mit HuggingFace-kompatiblem Interface."""
        # Filtere nur erlaubte Argumente für das Modell
        allowed_kwargs = {
            'input_ids', 'attention_mask', 'track_ids', 'bar_positions', 
            'labels', 'output_hidden_states', 'output_attentions'
        }
        filtered_kwargs = {k: v for k, v in kwargs.items() if k in allowed_kwargs}
        return self.model(**filtered_kwargs)
    
    def generate(self, **kwargs):
        """Generation mit HuggingFace-kompatiblem Interface."""
        return self.model.generate(**kwargs)
    
    def save_pretrained(self, save_directory: str):
        """Speichert Modell im HuggingFace-Format."""
        import os
        import json
        
        os.makedirs(save_directory, exist_ok=True)
        
        # Speichere Modell-Gewichte
        torch.save(self.state_dict(), os.path.join(save_directory, "pytorch_model.bin"))
        
        # Speichere Konfiguration
        with open(os.path.join(save_directory, "config.json"), "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)
    
    @classmethod
    def from_pretrained(cls, model_path: str):
        """Lädt Modell aus HuggingFace-Format."""
        import os
        import json
        
        # Lade Konfiguration
        with open(os.path.join(model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        config = MuseformerConfig.from_dict(config_dict)
        model = cls(config)
        
        # Lade Gewichte
        state_dict = torch.load(
            os.path.join(model_path, "pytorch_model.bin"),
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        
        return model 
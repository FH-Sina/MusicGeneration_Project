#!/usr/bin/env python3
"""
MusicGen Model Wrapper
Wrapper für AudioCraft MusicGen mit einheitlicher API
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .config import MusicGenConfig

class MusicGenWrapper(nn.Module):
    """
    Wrapper für MusicGen Model mit einheitlicher API
    Ähnlich der Museformer Architektur
    """
    
    def __init__(self, config: MusicGenConfig):
        super().__init__()
        self.config = config
        self.model = None
        self.tokenizer = None
        
    def load_pretrained(self):
        """Lädt das vortrainierte MusicGen Model"""
        try:
            from audiocraft.models import MusicGen
            self.model = MusicGen.get_pretrained(self.config.model_name)
            print(f"✅ MusicGen Model geladen: {self.config.model_name}")
        except ImportError:
            raise ImportError("AudioCraft nicht installiert! Installiere mit: pip install audiocraft")
    
    def forward(self, conditions, **kwargs):
        """Forward pass durch das Model"""
        if self.model is None:
            raise RuntimeError("Model nicht geladen! Rufe load_pretrained() auf.")
        
        return self.model.generate(conditions, **kwargs)
    
    def generate(self, prompts, **generation_params):
        """Generiert Audio basierend auf Text-Prompts"""
        if self.model is None:
            self.load_pretrained()
        
        # Merge generation parameters
        params = {**self.config.generation_params, **generation_params}
        
        # Set generation parameters
        self.model.set_generation_params(**params)
        
        # Generate audio
        with torch.no_grad():
            audio = self.model.generate(prompts)
        
        return audio
    
    def save_checkpoint(self, path: str, epoch: int, loss: float):
        """Speichert Model Checkpoint"""
        if self.model is None:
            raise RuntimeError("Model nicht geladen!")
        
        checkpoint = {
            'model_state_dict': self.model.lm.state_dict(),
            'config': self.config,
            'epoch': epoch,
            'loss': loss
        }
        
        torch.save(checkpoint, path)
        print(f"✅ Checkpoint gespeichert: {path}")
    
    def load_checkpoint(self, path: str):
        """Lädt Model Checkpoint"""
        if self.model is None:
            self.load_pretrained()
        
        checkpoint = torch.load(path, map_location='cpu')
        self.model.lm.load_state_dict(checkpoint['model_state_dict'])
        
        print(f"✅ Checkpoint geladen: {path}")
        return checkpoint.get('epoch', 0), checkpoint.get('loss', 0.0)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Gibt Model-Informationen zurück"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        return {
            "model_name": self.config.model_name,
            "sample_rate": self.config.sample_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_ratio": f"{trainable_params/1e6:.1f}M",
            "status": "loaded"
        }
    
    def set_training_mode(self, mode: bool = True):
        """Setzt Training/Evaluation Modus"""
        if self.model is None:
            raise RuntimeError("Model nicht geladen!")
        
        if mode:
            self.model.train()
            # Freeze compression model, only train language model
            for param in self.model.compression_model.parameters():
                param.requires_grad = False
            for param in self.model.lm.parameters():
                param.requires_grad = True
        else:
            self.model.eval()
    
    @property
    def device(self):
        """Gibt das Device des Models zurück"""
        if self.model is None:
            return torch.device('cpu')
        return next(self.model.parameters()).device
    
    def to(self, device):
        """Verschiebt Model auf Device"""
        if self.model is not None:
            self.model = self.model.to(device)
        return self

def create_musicgen_model(config: Optional[MusicGenConfig] = None) -> MusicGenWrapper:
    """Factory function für MusicGen Model"""
    if config is None:
        from .config import default_config
        config = default_config
    
    model = MusicGenWrapper(config)
    return model 
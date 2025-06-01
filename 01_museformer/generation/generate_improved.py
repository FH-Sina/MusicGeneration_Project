#!/usr/bin/env python3
"""
Verbessertes Generierungs-Script für Museformer
Verwendet Constrained Generation um sicherzustellen, dass Musik-Token generiert werden.
"""

import sys
sys.path.append('.')

import torch
import torch.nn.functional as F
from data.tokenizer import create_pop_midi_tokenizer
from model.config import get_config
from model.museformer import MuseformerForCausalLM
from pathlib import Path
import argparse
import numpy as np
from typing import List, Dict, Set

class ConstrainedMusicGenerator:
    """Generator mit Constraints für musikalische Token-Sequenzen."""
    
    def __init__(self, model_path: str, config_name: str = "ultra_small"):
        """
        Args:
            model_path: Pfad zum trainierten Modell
            config_name: Name der Konfiguration
        """
        self.tokenizer = create_pop_midi_tokenizer()
        self.config = get_config(config_name)
        self.config.vocab_size = self.tokenizer.vocab_size
        
        self.model = MuseformerForCausalLM.from_pretrained(model_path)
        self.model.eval()
        
        # Analysiere Token-Typen
        self._analyze_token_types()
        
        print(f"✓ Modell geladen: {sum(p.numel() for p in self.model.parameters()):,} Parameter")
        print(f"✓ Tokenizer: {self.tokenizer.vocab_size} Token")
        print(f"✓ Note-Token: {len(self.note_token_ids)}")
        print(f"✓ Velocity-Token: {len(self.vel_token_ids)}")
        print(f"✓ Duration-Token: {len(self.dur_token_ids)}")
    
    def _analyze_token_types(self):
        """Analysiert und kategorisiert alle Token-Typen."""
        self.note_token_ids = []
        self.vel_token_ids = []
        self.dur_token_ids = []
        self.shift_token_ids = []
        self.track_token_ids = []
        self.bar_token_ids = []
        
        for token_id, token_str in self.tokenizer.id_to_token.items():
            if token_str.startswith('NOTE_'):
                self.note_token_ids.append(token_id)
            elif token_str.startswith('VEL_'):
                self.vel_token_ids.append(token_id)
            elif token_str.startswith('DUR_'):
                self.dur_token_ids.append(token_id)
            elif token_str.startswith('SHIFT_'):
                self.shift_token_ids.append(token_id)
            elif token_str.startswith('TRACK_'):
                self.track_token_ids.append(token_id)
            elif token_str == '<BAR>':
                self.bar_token_ids.append(token_id)
    
    def generate_constrained(self, 
                           max_length: int = 200,
                           num_notes: int = 20,
                           temperature: float = 0.8,
                           seed: int = None) -> List[int]:
        """
        Generiert eine Token-Sequenz mit Constraints für musikalische Struktur.
        
        Args:
            max_length: Maximale Sequenz-Länge
            num_notes: Anzahl der zu generierenden Noten
            temperature: Sampling-Temperature
            seed: Random Seed
            
        Returns:
            Liste von Token-IDs
        """
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        
        input_ids = torch.tensor([[self.tokenizer.bos_token_id]], dtype=torch.long)
        generated_tokens = [self.tokenizer.bos_token_id]
        
        state = "need_track"
        notes_generated = 0
        
        for step in range(max_length):
            if notes_generated >= num_notes:
                break
                
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids)
                logits = outputs.logits[0, -1, :]
            
            # Erstelle Constraint-Mask
            mask = torch.full_like(logits, float('-inf'))
            
            if state == "need_track":
                # Erlaube Track-Token
                for tid in self.track_token_ids:
                    mask[tid] = 0
                next_state = "need_note"
                
            elif state == "need_note":
                # Erlaube Note-Token
                for tid in self.note_token_ids:
                    mask[tid] = 0
                # Gelegentlich erlaube auch Bar-Token
                if np.random.random() < 0.1 and len(generated_tokens) > 10:
                    for tid in self.bar_token_ids:
                        mask[tid] = 0
                next_state = "need_vel"
                
            elif state == "need_vel":
                # Erlaube Velocity-Token
                for tid in self.vel_token_ids:
                    mask[tid] = 0
                next_state = "need_dur"
                
            elif state == "need_dur":
                # Erlaube Duration-Token
                for tid in self.dur_token_ids:
                    mask[tid] = 0
                next_state = "need_shift_or_note"
                
            elif state == "need_shift_or_note":
                # Erlaube Shift-Token oder direkt neue Note
                for tid in self.shift_token_ids:
                    mask[tid] = 0
                for tid in self.note_token_ids:
                    mask[tid] = 0
                # Gelegentlich erlaube Bar-Token
                if np.random.random() < 0.15:
                    for tid in self.bar_token_ids:
                        mask[tid] = 0
                next_state = "decide_next"
            
            # Sample mit Constraints
            next_token = self._constrained_sample(logits, mask, temperature)
            
            # Update Zustand
            token_str = self.tokenizer.id_to_token.get(next_token, '')
            
            if state == "need_shift_or_note":
                if next_token in self.note_token_ids:
                    state = "need_vel"
                    notes_generated += 1
                elif next_token in self.shift_token_ids:
                    state = "need_note"
                elif next_token in self.bar_token_ids:
                    state = "need_note"
                else:
                    state = "need_note"
            elif state == "decide_next":
                if next_token in self.shift_token_ids:
                    state = "need_note"
                elif next_token in self.note_token_ids:
                    state = "need_vel"
                    notes_generated += 1
                elif next_token in self.bar_token_ids:
                    state = "need_note"
                else:
                    state = "need_note"
            elif state == "need_note" and next_token in self.bar_token_ids:
                state = "need_note"  # Nach Bar kommt Note
            else:
                state = next_state
                if next_token in self.note_token_ids:
                    notes_generated += 1
            
            # Füge Token hinzu
            generated_tokens.append(next_token)
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]])], dim=-1)
            
            # Stoppe bei EOS
            if next_token == self.tokenizer.eos_token_id:
                break
        
        return generated_tokens
    
    def _constrained_sample(self, logits: torch.Tensor, mask: torch.Tensor, temperature: float) -> int:
        """Sicheres Sampling mit Constraints."""
        try:
            constrained_logits = logits + mask
            probs = F.softmax(constrained_logits / temperature, dim=-1)
            
            # Überprüfe auf gültige Wahrscheinlichkeiten
            if torch.isnan(probs).any() or torch.isinf(probs).any() or (probs < 0).any():
                return torch.argmax(constrained_logits).item()
            else:
                return torch.multinomial(probs, num_samples=1).item()
        except Exception:
            return torch.argmax(constrained_logits).item()
    
    def generate_midi(self, 
                     output_path: str,
                     max_length: int = 200,
                     num_notes: int = 20,
                     temperature: float = 0.8,
                     seed: int = None) -> bool:
        """
        Generiert MIDI-File mit Constrained Generation.
        
        Args:
            output_path: Ausgabe-Pfad für MIDI-File
            max_length: Maximale Token-Sequenz-Länge
            num_notes: Anzahl der zu generierenden Noten
            temperature: Sampling-Temperature
            seed: Random Seed
            
        Returns:
            True wenn erfolgreich, False sonst
        """
        print(f"🎵 Generiere MIDI mit {num_notes} Noten...")
        
        # Generiere Token-Sequenz
        tokens = self.generate_constrained(
            max_length=max_length,
            num_notes=num_notes,
            temperature=temperature,
            seed=seed
        )
        
        # Analysiere generierte Token
        token_types = {}
        for token in tokens:
            token_str = self.tokenizer.id_to_token.get(token, f'<UNK_{token}>')
            token_type = token_str.split('_')[0] if '_' in token_str else token_str
            token_types[token_type] = token_types.get(token_type, 0) + 1
        
        print(f"   Generierte Token-Typen: {token_types}")
        
        # Konvertiere zu MIDI
        try:
            midi = self.tokenizer.tokens_to_midi(tokens, output_path)
            
            note_count = sum(len(inst.notes) for inst in midi.instruments)
            duration = midi.get_end_time()
            
            print(f"   MIDI erstellt:")
            print(f"     Datei: {output_path}")
            print(f"     Instrumente: {len(midi.instruments)}")
            print(f"     Noten: {note_count}")
            print(f"     Dauer: {duration:.2f}s")
            
            return note_count > 0
            
        except Exception as e:
            print(f"❌ MIDI-Konvertierung Fehler: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Verbessertes Museformer MIDI Generation')
    parser.add_argument('--model_path', type=str, default='outputs/best_model',
                       help='Pfad zum trainierten Modell')
    parser.add_argument('--output_dir', type=str, default='generated_music_improved',
                       help='Ausgabe-Verzeichnis')
    parser.add_argument('--num_samples', type=int, default=3,
                       help='Anzahl der zu generierenden Samples')
    parser.add_argument('--num_notes', type=int, default=20,
                       help='Anzahl der Noten pro Sample')
    parser.add_argument('--max_length', type=int, default=200,
                       help='Maximale Token-Sequenz-Länge')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling Temperature')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random Seed')
    
    args = parser.parse_args()
    
    # Erstelle Ausgabe-Verzeichnis
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Initialisiere Generator
    generator = ConstrainedMusicGenerator(args.model_path)
    
    # Generiere Samples
    successful_samples = 0
    
    for i in range(args.num_samples):
        print(f"\n🎼 Generiere Sample {i+1}/{args.num_samples}")
        
        output_path = output_dir / f"generated_sample_{i+1:03d}.mid"
        seed = args.seed + i if args.seed is not None else None
        
        success = generator.generate_midi(
            output_path=str(output_path),
            max_length=args.max_length,
            num_notes=args.num_notes,
            temperature=args.temperature,
            seed=seed
        )
        
        if success:
            successful_samples += 1
            print(f"✅ Sample {i+1} erfolgreich generiert")
        else:
            print(f"❌ Sample {i+1} fehlgeschlagen")
    
    print(f"\n🎉 Generierung abgeschlossen!")
    print(f"   Erfolgreiche Samples: {successful_samples}/{args.num_samples}")
    print(f"   Ausgabe-Verzeichnis: {output_dir}")

if __name__ == "__main__":
    main() 
#!/usr/bin/env python3
"""
MusicGen Generation mit trainiertem Model auf POP909
"""

import argparse
import torch
from pathlib import Path
import torchaudio
import json

def load_trained_model(checkpoint_path):
    """Lädt das trainierte MusicGen Model."""
    
    print(f"Lade trainiertes Model: {checkpoint_path}")
    
    try:
        from audiocraft.models import MusicGen
    except ImportError:
        print("FEHLER: AudioCraft nicht installiert!")
        return None
    
    # Lade Pre-trained Model
    model = MusicGen.get_pretrained('facebook/musicgen-small')
    
    # Lade trainierte Weights
    if Path(checkpoint_path).exists():
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.lm.load_state_dict(checkpoint['model_state_dict'])
        print(f"Trainierte Weights geladen (Epoche {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"WARNUNG: Checkpoint nicht gefunden: {checkpoint_path}")
        print("Verwende Pre-trained Model")
    
    return model

def generate_music_trained(checkpoint_path, num_samples=5, duration=30.0, output_dir="outputs/generated_trained"):
    """Generiert Musik mit trainiertem MusicGen Model."""
    
    # Lade trainiertes Model
    model = load_trained_model(checkpoint_path)
    if model is None:
        return False
    
    model.set_generation_params(duration=duration)
    
    # Pop-Music Prompts (ähnlich zu POP909 Training-Daten)
    prompts = [
        "pop music with piano melody",
        "upbeat pop song with drums and bass",
        "melodic pop ballad with guitar",
        "modern pop music with synthesizer",
        "catchy pop tune with electronic elements",
        "energetic pop song with strong rhythm",
        "romantic pop ballad with soft vocals",
        "dance pop music with electronic beats"
    ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = []
    
    print(f"\nGeneriere {num_samples} Samples mit trainiertem Model...")
    
    for i, prompt in enumerate(prompts[:num_samples]):
        print(f"\nGeneriere Sample {i+1}/{num_samples}")
        print(f"Prompt: '{prompt}'")
        
        try:
            # Generiere Audio
            with torch.no_grad():
                wav = model.generate([prompt])
            
            # Speichere Audio
            output_file = output_path / f"trained_sample_{i+1:03d}.wav"
            torchaudio.save(str(output_file), wav[0].cpu(), model.sample_rate)
            
            results.append({
                "file": str(output_file),
                "prompt": prompt,
                "duration": duration,
                "sample_rate": model.sample_rate,
                "model": "trained_musicgen_pop909"
            })
            
            print(f"✅ Gespeichert: {output_file}")
            
        except Exception as e:
            print(f"❌ Fehler bei Sample {i+1}: {e}")
            continue
    
    # Speichere Metadata
    metadata_file = output_path / "trained_generation_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n🎉 Generierung abgeschlossen!")
    print(f"{len(results)} Samples erstellt in: {output_path}")
    print(f"Metadata: {metadata_file}")
    
    return True

def compare_models(num_samples=3, duration=15.0):
    """Vergleicht Pre-trained vs Trained Model."""
    
    print("🔄 Vergleiche Pre-trained vs Trained Model...")
    
    # Pre-trained Model
    print("\n1. Pre-trained Model:")
    from audiocraft.models import MusicGen
    pretrained_model = MusicGen.get_pretrained('facebook/musicgen-small')
    pretrained_model.set_generation_params(duration=duration)
    
    # Trained Model
    print("\n2. Trained Model:")
    checkpoint_path = "outputs/checkpoints/musicgen_working_epoch_5.pt"
    trained_model = load_trained_model(checkpoint_path)
    if trained_model:
        trained_model.set_generation_params(duration=duration)
    
    # Test Prompts
    test_prompts = [
        "pop music with piano",
        "upbeat pop song",
        "melodic pop ballad"
    ]
    
    output_dir = Path("outputs/comparison")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(test_prompts[:num_samples]):
        print(f"\nTest {i+1}: '{prompt}'")
        
        # Pre-trained
        try:
            with torch.no_grad():
                wav_pretrained = pretrained_model.generate([prompt])
            pretrained_file = output_dir / f"pretrained_{i+1:03d}.wav"
            torchaudio.save(str(pretrained_file), wav_pretrained[0].cpu(), pretrained_model.sample_rate)
            print(f"✅ Pre-trained: {pretrained_file}")
        except Exception as e:
            print(f"❌ Pre-trained error: {e}")
        
        # Trained
        if trained_model:
            try:
                with torch.no_grad():
                    wav_trained = trained_model.generate([prompt])
                trained_file = output_dir / f"trained_{i+1:03d}.wav"
                torchaudio.save(str(trained_file), wav_trained[0].cpu(), trained_model.sample_rate)
                print(f"✅ Trained: {trained_file}")
            except Exception as e:
                print(f"❌ Trained error: {e}")
    
    print(f"\n🎯 Vergleich abgeschlossen in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description='MusicGen Generation mit trainiertem Model')
    parser.add_argument('--checkpoint', type=str, 
                       default='outputs/checkpoints/musicgen_working_epoch_5.pt',
                       help='Pfad zum trainierten Model Checkpoint')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Anzahl der zu generierenden Samples')
    parser.add_argument('--duration', type=float, default=30.0,
                       help='Dauer der generierten Samples')
    parser.add_argument('--output_dir', type=str, default='outputs/generated_trained',
                       help='Ausgabe-Verzeichnis')
    parser.add_argument('--compare', action='store_true',
                       help='Vergleiche Pre-trained vs Trained Model')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models(num_samples=3, duration=15.0)
    else:
        generate_music_trained(
            checkpoint_path=args.checkpoint,
            num_samples=args.num_samples,
            duration=args.duration,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main() 
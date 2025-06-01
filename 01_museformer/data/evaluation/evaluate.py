#!/usr/bin/env python3
"""
Museformer Evaluation Script
Evaluiert generierte MIDI-Dateien mit verschiedenen Metriken
"""

import argparse
import numpy as np
import pretty_midi
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from scipy import stats
from .metrics import *

def extract_midi_features(midi_file):
    """Extrahiert Features aus einer MIDI-Datei."""
    
    try:
        # Lade MIDI
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        
        # Grundlegende Features
        features = {
            "duration": midi.get_end_time(),
            "num_instruments": len(midi.instruments),
            "total_notes": sum(len(inst.notes) for inst in midi.instruments),
        }
        
        # Note-basierte Features
        all_notes = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                all_notes.extend(instrument.notes)
        
        if all_notes:
            pitches = [note.pitch for note in all_notes]
            velocities = [note.velocity for note in all_notes]
            durations = [note.end - note.start for note in all_notes]
            
            features.update({
                "pitch_mean": float(np.mean(pitches)),
                "pitch_std": float(np.std(pitches)),
                "pitch_range": float(max(pitches) - min(pitches)),
                "velocity_mean": float(np.mean(velocities)),
                "velocity_std": float(np.std(velocities)),
                "note_duration_mean": float(np.mean(durations)),
                "note_duration_std": float(np.std(durations)),
                "notes_per_second": len(all_notes) / features["duration"] if features["duration"] > 0 else 0
            })
        
        # Rhythmus-Features
        onset_times = [note.start for note in all_notes]
        if len(onset_times) > 1:
            inter_onset_intervals = np.diff(sorted(onset_times))
            features.update({
                "rhythm_regularity": float(1.0 / (np.std(inter_onset_intervals) + 1e-6)),
                "avg_inter_onset": float(np.mean(inter_onset_intervals))
            })
        
        return features
        
    except Exception as e:
        print(f"Fehler bei {midi_file}: {e}")
        return None

def analyze_directory(directory, label="unknown"):
    """Analysiert alle MIDI-Dateien in einem Verzeichnis."""
    
    directory = Path(directory)
    midi_files = list(directory.glob("*.mid")) + list(directory.glob("*.midi"))
    
    if not midi_files:
        print(f"Keine MIDI-Dateien in {directory} gefunden")
        return []
    
    print(f"Analysiere {len(midi_files)} MIDI-Dateien in {directory}...")
    
    results = []
    for midi_file in midi_files:
        print(f"  Analysiere: {midi_file.name}")
        features = extract_midi_features(midi_file)
        if features:
            features["file"] = str(midi_file)
            features["label"] = label
            results.append(features)
    
    return results

def compare_generated_vs_original():
    """Vergleicht generierte vs. originale MIDI-Dateien."""
    
    print("🔍 Analysiere Museformer Generierung...")
    
    # Analysiere generierte MIDI-Dateien
    generated_results = []
    generated_dir = Path("outputs/generated")
    if generated_dir.exists():
        generated_files = list(generated_dir.glob("*.mid"))
        for file in generated_files:
            features = extract_midi_features(file)
            if features:
                features["model"] = "generated"
                features["file"] = str(file)
                generated_results.append(features)
    
    # Analysiere Original POP909 MIDI-Dateien (Sample)
    original_results = []
    original_dir = Path("data/midi/samples")
    if original_dir.exists():
        original_files = list(original_dir.glob("*.mid"))[:10]  # Nur erste 10 für Vergleich
        for file in original_files:
            features = extract_midi_features(file)
            if features:
                features["model"] = "original"
                features["file"] = str(file)
                original_results.append(features)
    
    # Kombiniere Ergebnisse
    all_results = generated_results + original_results
    
    if not all_results:
        print("Keine MIDI-Dateien zum Analysieren gefunden!")
        return
    
    # Erstelle DataFrame
    df = pd.DataFrame(all_results)
    
    # Speichere Ergebnisse
    output_dir = Path("evaluation/results")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "midi_features_comparison.csv", index=False)
    
    # Statistischer Vergleich
    print("\n📊 Statistischer Vergleich:")
    
    key_features = ["pitch_mean", "velocity_mean", "note_duration_mean", "notes_per_second"]
    
    comparison_stats = {}
    for feature in key_features:
        if feature in df.columns:
            generated_values = df[df["model"] == "generated"][feature].values
            original_values = df[df["model"] == "original"][feature].values
            
            if len(generated_values) > 0 and len(original_values) > 0:
                # T-Test
                t_stat, p_value = stats.ttest_ind(generated_values, original_values)
                
                comparison_stats[feature] = {
                    "generated_mean": float(np.mean(generated_values)),
                    "generated_std": float(np.std(generated_values)),
                    "original_mean": float(np.mean(original_values)),
                    "original_std": float(np.std(original_values)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05)
                }
                
                print(f"\n{feature}:")
                print(f"  Generated: {np.mean(generated_values):.4f} ± {np.std(generated_values):.4f}")
                print(f"  Original:  {np.mean(original_values):.4f} ± {np.std(original_values):.4f}")
                print(f"  p-value:   {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Speichere Statistiken
    with open(output_dir / "comparison_statistics.json", 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    
    # Erstelle Visualisierung
    create_comparison_plots(df, output_dir)
    
    print(f"\n✅ Evaluation abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {output_dir}")

def create_comparison_plots(df, output_dir):
    """Erstellt Vergleichs-Plots."""
    
    key_features = ["pitch_mean", "velocity_mean", "note_duration_mean", "notes_per_second"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(key_features):
        if feature in df.columns and i < len(axes):
            ax = axes[i]
            
            # Box Plot
            generated_data = df[df["model"] == "generated"][feature].values
            original_data = df[df["model"] == "original"][feature].values
            
            ax.boxplot([generated_data, original_data], 
                      labels=["Generated", "Original"])
            ax.set_title(f"{feature}")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "midi_feature_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plots gespeichert: {output_dir}/midi_feature_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluiere generierte MIDI-Dateien')
    parser.add_argument('--compare', action='store_true',
                       help='Vergleiche Generated vs Original MIDI')
    parser.add_argument('--directory', type=str,
                       help='Analysiere spezifisches Verzeichnis')
    parser.add_argument('--label', type=str, default='unknown',
                       help='Label für die Analyse')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_generated_vs_original()
    elif args.directory:
        results = analyze_directory(args.directory, args.label)
        if results:
            output_file = Path("evaluation/results") / f"{args.label}_features.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Ergebnisse gespeichert: {output_file}")
    else:
        print("Verwende --compare oder --directory")

if __name__ == "__main__":
    main() 
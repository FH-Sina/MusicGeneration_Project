#!/usr/bin/env python3
"""
Evaluation Script für generierte Musik
Analysiert Audio-Features von MusicGen Outputs
"""

import argparse
import numpy as np
import librosa
import matplotlib.pyplot as plt
from pathlib import Path
import json
import pandas as pd
from scipy import stats

def extract_audio_features(audio_file):
    """Extrahiert Audio-Features aus einer WAV-Datei."""
    
    try:
        # Lade Audio
        y, sr = librosa.load(audio_file, sr=32000)
        
        # Grundlegende Features
        features = {
            "duration": len(y) / sr,
            "sample_rate": sr,
            "rms_energy": float(np.mean(librosa.feature.rms(y=y))),
            "zero_crossing_rate": float(np.mean(librosa.feature.zero_crossing_rate(y))),
        }
        
        # Spektrale Features
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        
        features.update({
            "spectral_centroid_mean": float(np.mean(spectral_centroids)),
            "spectral_centroid_std": float(np.std(spectral_centroids)),
            "spectral_rolloff_mean": float(np.mean(spectral_rolloff)),
            "spectral_bandwidth_mean": float(np.mean(spectral_bandwidth)),
        })
        
        # MFCC Features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        for i in range(13):
            features[f"mfcc_{i+1}_mean"] = float(np.mean(mfccs[i]))
            features[f"mfcc_{i+1}_std"] = float(np.std(mfccs[i]))
        
        # Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])
        
        # Chroma Features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        features["chroma_mean"] = float(np.mean(chroma))
        features["chroma_std"] = float(np.std(chroma))
        
        return features
        
    except Exception as e:
        print(f"Fehler bei {audio_file}: {e}")
        return None

def analyze_directory(directory, label="unknown"):
    """Analysiert alle WAV-Dateien in einem Verzeichnis."""
    
    directory = Path(directory)
    wav_files = list(directory.glob("*.wav"))
    
    if not wav_files:
        print(f"Keine WAV-Dateien in {directory} gefunden")
        return []
    
    print(f"Analysiere {len(wav_files)} Dateien in {directory}...")
    
    results = []
    for wav_file in wav_files:
        print(f"  Analysiere: {wav_file.name}")
        features = extract_audio_features(wav_file)
        if features:
            features["file"] = str(wav_file)
            features["label"] = label
            results.append(features)
    
    return results

def compare_models():
    """Vergleicht Pre-trained vs Trained Model Outputs."""
    
    print("🔍 Analysiere generierte Musik...")
    
    # Analysiere Pre-trained Model Outputs
    pretrained_results = []
    comparison_dir = Path("outputs/comparison")
    if comparison_dir.exists():
        pretrained_files = list(comparison_dir.glob("pretrained_*.wav"))
        for file in pretrained_files:
            features = extract_audio_features(file)
            if features:
                features["model"] = "pretrained"
                features["file"] = str(file)
                pretrained_results.append(features)
    
    # Analysiere Trained Model Outputs
    trained_results = []
    if comparison_dir.exists():
        trained_files = list(comparison_dir.glob("trained_*.wav"))
        for file in trained_files:
            features = extract_audio_features(file)
            if features:
                features["model"] = "trained"
                features["file"] = str(file)
                trained_results.append(features)
    
    # Kombiniere Ergebnisse
    all_results = pretrained_results + trained_results
    
    if not all_results:
        print("Keine Dateien zum Analysieren gefunden!")
        return
    
    # Erstelle DataFrame
    df = pd.DataFrame(all_results)
    
    # Speichere Ergebnisse
    output_dir = Path("outputs/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    df.to_csv(output_dir / "audio_features_comparison.csv", index=False)
    
    # Statistischer Vergleich
    print("\n📊 Statistischer Vergleich:")
    
    key_features = ["rms_energy", "spectral_centroid_mean", "tempo", "chroma_mean"]
    
    comparison_stats = {}
    for feature in key_features:
        if feature in df.columns:
            pretrained_values = df[df["model"] == "pretrained"][feature].values
            trained_values = df[df["model"] == "trained"][feature].values
            
            if len(pretrained_values) > 0 and len(trained_values) > 0:
                # T-Test
                t_stat, p_value = stats.ttest_ind(pretrained_values, trained_values)
                
                comparison_stats[feature] = {
                    "pretrained_mean": float(np.mean(pretrained_values)),
                    "pretrained_std": float(np.std(pretrained_values)),
                    "trained_mean": float(np.mean(trained_values)),
                    "trained_std": float(np.std(trained_values)),
                    "t_statistic": float(t_stat),
                    "p_value": float(p_value),
                    "significant": bool(p_value < 0.05)
                }
                
                print(f"\n{feature}:")
                print(f"  Pre-trained: {np.mean(pretrained_values):.4f} ± {np.std(pretrained_values):.4f}")
                print(f"  Trained:     {np.mean(trained_values):.4f} ± {np.std(trained_values):.4f}")
                print(f"  p-value:     {p_value:.4f} {'(significant)' if p_value < 0.05 else '(not significant)'}")
    
    # Speichere Statistiken
    with open(output_dir / "comparison_statistics.json", 'w') as f:
        json.dump(comparison_stats, f, indent=2)
    
    # Erstelle Visualisierung
    create_comparison_plots(df, output_dir)
    
    print(f"\n✅ Evaluation abgeschlossen!")
    print(f"Ergebnisse gespeichert in: {output_dir}")

def create_comparison_plots(df, output_dir):
    """Erstellt Vergleichs-Plots."""
    
    key_features = ["rms_energy", "spectral_centroid_mean", "tempo", "chroma_mean"]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(key_features):
        if feature in df.columns and i < len(axes):
            ax = axes[i]
            
            # Box Plot
            pretrained_data = df[df["model"] == "pretrained"][feature].values
            trained_data = df[df["model"] == "trained"][feature].values
            
            ax.boxplot([pretrained_data, trained_data], 
                      labels=["Pre-trained", "Trained"])
            ax.set_title(f"{feature}")
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "feature_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📈 Plots gespeichert: {output_dir}/feature_comparison.png")

def main():
    parser = argparse.ArgumentParser(description='Evaluiere generierte Musik')
    parser.add_argument('--compare', action='store_true',
                       help='Vergleiche Pre-trained vs Trained Model')
    parser.add_argument('--directory', type=str,
                       help='Analysiere spezifisches Verzeichnis')
    parser.add_argument('--label', type=str, default='unknown',
                       help='Label für die Analyse')
    
    args = parser.parse_args()
    
    if args.compare:
        compare_models()
    elif args.directory:
        results = analyze_directory(args.directory, args.label)
        if results:
            output_file = Path("outputs/evaluation") / f"{args.label}_features.json"
            output_file.parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Ergebnisse gespeichert: {output_file}")
    else:
        print("Verwende --compare oder --directory")

if __name__ == "__main__":
    main() 
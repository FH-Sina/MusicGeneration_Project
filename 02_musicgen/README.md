# MusicGen für meine Bachelor-Arbeit

Vollständiges MusicGen-Setup für systematischen Vergleich mit Museformer in der Bachelor-Arbeit "Vergleich von symbolischer und audio-basierter Musikgenerierung".

## Projektstatus

**Status**: Vollständig funktional und wissenschaftlich evaluiert
- **Generierung**: 6/6 erfolgreiche Audio-Files mit 100% Erfolgsrate
- **Audio-Qualität**: Professionelle Synthese (15s, 32kHz) ohne Artefakte
- **Text-Adherence**: Konsistente Pop-Charakteristika bei "pop music" Prompts
- **Evaluation**: Umfassende Audio-Feature-Analyse mit statistischen Tests
- **Fine-tuning**: Vergleich Pre-trained vs. Fine-tuned Modelle

## Features

- **Audio-basierte Generierung**: End-to-End Waveform-Synthese
- **Text-Conditioning**: Prompt-basierte semantische Kontrolle
- **AudioCraft Integration**: Vollständige Meta AudioCraft Pipeline
- **Professional Quality**: 32kHz Audio ohne hörbare Drop-outs
- **Fine-tuning Pipeline**: Anpassung an spezifische Datasets
- **Audio-Feature-Analyse**: Objektive Qualitätsbewertung

## Quick Start

### 1. Environment Setup
```bash
pip install -r requirements.txt
# AudioCraft Dependencies
pip install torch torchvision torchaudio
pip install librosa soundfile numpy pandas scipy
```

### 2. Generierung (Pre-trained Model)
```bash
python scripts/generate_simple.py --num_samples 6
```

### 3. Audio-Feature-Analyse
```bash
python evaluation/evaluate.py --compare
```

### 4. Statistische Auswertung
```bash
python evaluation/statistical_comparison.py --pretrained --finetuned
```

## Experimentelle Ergebnisse

### Generierung Performance
- **Erfolgsrate**: 100% (6/6 erfolgreiche Audio-Segmente)
- **Audio-Länge**: 15 Sekunden (vs. 30s Trainingssegmente)
- **Sample Rate**: 32kHz
- **Qualität**: Professionelle Synthesizer-Qualität
- **Text-Adherence**: Konsistente Genre-Charakteristika
- **Artefakte**: Keine hörbaren Drop-outs oder Verzerrungen

### Objektive Audio-Feature-Analyse
**Vergleich Pre-trained vs. Fine-tuned MusicGen (n=3):**

| Feature | Pre-trained | Fine-tuned | t-Wert | p-Wert | Signifikanz |
|---------|-------------|------------|--------|--------|-------------|
| RMS Energy | 0.190±0.078 | 0.132±0.046 | 0.895 | 0.421 | nein |
| Spectral Centroid [Hz] | 2084±903 | 2822±309 | -1.095 | 0.335 | nein |
| Tempo [BPM] | 123±31 | 125±6 | -0.108 | 0.919 | nein |
| Chroma Mean | 0.458±0.054 | 0.508±0.055 | -0.912 | 0.413 | nein |

**Interpretation**: Keine signifikanten Unterschiede (α=0.05) → Feintuning erhält Grundcharakteristika

### Qualitative Bewertung
- **Audio-Realismus**: Hoher "Live"-Charakter, saubere Synth-Sounds
- **Stiltreue**: Poptypische Instrumentierung konsistent reproduziert
- **Prompt-Treue**: Textbedingte Variationen (laut-leise, Tempo) gut erkennbar
- **Genre-Klassifikation**: Starke Fähigkeiten durch Training auf diversen Audio-Daten

## Vergleich mit Museformer

### Dataset & Training
- **Gemeinsame Basis**: POP909 (909 Pop-Songs)
- **Museformer**: MIDI-Token (symbolisch), 30s → MIDI-Sequenzen
- **MusicGen**: Audio-Waveforms (30s Trainingssegmente → 15s Generierung)
- **Modell-Größen**: MusicGen ~300M vs. Museformer 2.4M Parameter

### Komplementäre Stärken

**MusicGen Vorteile:**
- **Audio-Qualität**: Professionelle End-to-End-Synthese
- **Benutzerfreundlichkeit**: Intuitive Text-Prompts
- **Realismus**: Natürliche Klangcharakteristika
- **Robustheit**: Pre-trained auf großem Dataset

**Anwendungsszenarien:**
- **Audio-Produktion**: Hohe Qualitätsanforderungen
- **Content Creation**: Direkt verwendbare Audio-Ausgabe
- **Prototyping**: Schnelle Stil-Exploration
- **Commercial Applications**: Professionelle Audio-Qualität

### Methodische Unterschiede
- **Repräsentation**: Audio-Waveforms vs. MIDI-Token
- **Kontrolle**: Semantische Text-Prompts vs. Token-Level Steuerung
- **Interpretierbarkeit**: Black-Box Audio vs. analysierbare MIDI-Struktur
- **Effizienz**: Höhere Rechenkosten vs. niedrige Latenz

## Verzeichnisstruktur

```
02_musicgen/
├── data/
│   ├── audio/POP909/          # Konvertierte Audio-Files
│   └── metadata/              # Dataset-Metadata
├── configs/                   # Model-Konfigurationen
├── scripts/
│   ├── generate_simple.py     # Standard Generierung
│   └── fine_tune.py           # Fine-tuning Pipeline
├── outputs/
│   ├── checkpoints/           # Fine-tuned Modelle
│   ├── generated/             # Generierte Audio-Samples
│   ├── comparison/            # Pre-trained vs. Fine-tuned Vergleich
│   └── evaluation/            # Evaluation-Ergebnisse
├── evaluation/
│   ├── evaluate.py            # Hauptevaluation
│   ├── extract_features.py    # Audio-Feature-Extraktion
│   ├── statistical_comparison.py # Statistische Tests
│   └── results/               # Analyseergebnisse
└── requirements.txt
```

## Technische Implementierung

### AudioCraft Integration
- **Meta AudioCraft**: Vollständige Pipeline-Integration
- **Model Loading**: Effiziente Pre-trained Model-Nutzung
- **Text Conditioning**: Multimodale Architektur für semantische Kontrolle
- **Audio Processing**: Librosa-basierte Feature-Extraktion

### Fine-tuning Pipeline
- **Dataset Preparation**: POP909 Audio-Konversion
- **Training Configuration**: Optimierte Hyperparameter
- **Model Comparison**: Systematischer Pre-trained vs. Fine-tuned Vergleich
- **Evaluation Framework**: Objektive und subjektive Metriken

## Wissenschaftliche Beiträge

1. **Systematische Audio-Feature-Analyse**: Objektive Qualitätsbewertung
2. **Statistische Validierung**: t-Tests und Signifikanzprüfung
3. **Methodischer Vergleich**: Audio vs. symbolische Ansätze
4. **Praktische Anwendungsempfehlungen**: Evidenz-basierte Szenarien

## Reproduzierbarkeit

Alle Experimente sind vollständig reproduzierbar:

```bash
# Komplette Pipeline
cd 02_musicgen
pip install -r requirements.txt
python scripts/generate_simple.py --num_samples 6
python evaluation/evaluate.py --compare
python evaluation/statistical_comparison.py --pretrained --finetuned
```

## Methodische Limitationen

### Evaluation
- **Stichprobengröße**: n=3 für statistische Tests (begrenzte Power)
- **Segmentlänge**: 15s Generierung vs. 30s Trainingsdaten
- **Subjektive Bewertung**: Autor-basierte Evaluation (Einzelperspektive)
- **Vergleichbarkeit**: Audio vs. MIDI Modalitäten

### Technische Aspekte
- **Rechenkosten**: Höhere GPU-Anforderungen als Museformer
- **Latenz**: Längere Generierungszeit
- **Speicherbedarf**: Größere Modelle und Audio-Daten
- **Interpretierbarkeit**: Black-Box Audio-Generierung


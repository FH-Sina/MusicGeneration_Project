# Museformer für meine Bachelorarbeit

Museformer ist ein Transformer-basiertes Modell für symbolische Musikgenerierung, das auf MIDI-Daten trainiert wird. Diese Implementierung wurde erfolgreich für die Bachelor-Arbeit "Vergleich von symbolischer und audio-basierter Musikgenerierung" entwickelt.

## Projektstatus

**Status**: Vollständig funktional und wissenschaftlich evaluiert
- **Training**: 76k Steps erfolgreich abgeschlossen
- **Generierung**: 3/3 erfolgreiche MIDI-Files mit 100% Validität
- **Innovation**: Constrained Generation für musikalische Kohärenz implementiert
- **Evaluation**: Umfassende Metriken und Vergleichsanalyse

## Features

- **Symbolische MIDI-Generierung**: Token-basierte Musikgenerierung mit NOTE_ON/OFF, VELOCITY, DURATION, SHIFT
- **Constrained Generation**: State Machine für musikalische Kohärenz und Phrasenstrukturen
- **POP909 Dataset**: Training auf 909 Pop-Songs für Genre-spezifische Generierung
- **Effiziente Architektur**: 2.4M Parameter für optimale GPU-Memory-Nutzung (6GB RTX 3060)
- **Token Balance**: Ausgewogene Repräsentation aller musikalischen Elemente
- **Phrase Boundaries**: Logische Abschlüsse und strukturelle Konsistenz

## Struktur

```
01_museformer/
├── model/
│   ├── museformer.py          # Hauptmodell-Implementierung
│   ├── attention.py           # Bar-Level Attention Mechanismen
│   ├── embedding.py           # MIDI Token Embeddings
│   └── config.py              # Modell-Konfiguration
├── data/
│   ├── midi_processor.py      # MIDI Preprocessing
│   ├── tokenizer.py           # MIDI Tokenization
│   ├── dataset.py             # Dataset-Klassen
│   └── midi/                  # POP909 MIDI Dataset (909 Songs)
├── training/
│   ├── train.py               # Training Script
│   ├── trainer.py             # Trainer-Klasse
│   └── utils.py               # Training Utilities
├── generation/
│   ├── generate.py            # Standard Generierung
│   ├── generate_constrained.py # Constrained Generation (Hauptmethode)
│   └── postprocess.py         # MIDI Post-Processing
├── evaluation/
│   ├── metrics.py             # Evaluations-Metriken (MIDI-spezifisch)
│   ├── evaluate.py            # Evaluation Pipeline
│   └── analysis.py            # Musik-Analyse Tools
├── configs/
│   └── ultra_small.yaml       # Optimierte Konfiguration
└── outputs/
    └── generated/             # Generierte MIDI-Files
```

## Experimentelle Ergebnisse

### Training Performance
- **Modellgröße**: 2.4M Parameter
- **Training Zeit**: 2-3 Stunden auf RTX 3060
- **Final Loss**: ~2.5
- **Training Steps**: 76k Steps
- **Memory Usage**: Optimiert für 6GB GPU

### Generierung Performance
- **Erfolgsrate**: 100% (3/3 erfolgreiche MIDI-Sequenzen)
- **Token-Verteilung**: 14-15 Noten pro 256-Token-Fenster
- **Validität**: Fehlerfreie Token-Dekodierung und MIDI-Abspielbarkeit
- **Kohärenz**: Phrase-Boundaries und Wiederholungsstrukturen
- **Token-Balance**: Alle musikalischen Elemente (NOTE_ON/OFF, VELOCITY, DURATION, SHIFT) vertreten

### Qualitative Bewertung
- **Melodieführung**: Klar und nachvollziehbar, repetitive Motive
- **Rhythmus**: Stabiles, metronomisches Timing
- **Harmonie**: Einfache, funktionale Akkordfolgen (POP909-typisch)
- **Strukturelle Klarheit**: Präzise melodische Kontrolle durch symbolische Repräsentation

## Usage

### 1. Environment Setup
```bash
pip install torch torchvision torchaudio
pip install pretty_midi librosa numpy pandas
pip install transformers accelerate
```

### 2. Data Preparation
```bash
# POP909 Dataset bereits im data/midi/ Verzeichnis
python data/midi_processor.py --input_dir data/midi --output_dir data/processed
```

### 3. Training
```bash
python training/train.py --config configs/ultra_small.yaml
```

### 4. Constrained Generation (Empfohlen)
```bash
python generation/generate_constrained.py --num_samples 3
```

### 5. Evaluation
```bash
python evaluation/evaluate.py --compare
```

## Technische Innovationen

### Constrained Generation
- **State Machine**: Gewährleistet musikalische Token-Sequenzen
- **Phrase Boundaries**: Verhindert abrupte Schnitte
- **Wiederholungsstrukturen**: Musikalische Kohärenz
- **Token-Balancing**: Ausgewogene Generierung aller Elementtypen

### Bug Fixes
- **Vocab-Size Error**: Kritischer Fehler in Model-Initialisierung behoben
- **CUDA Indexing**: Memory-optimierte Token-Verarbeitung
- **Generation Pipeline**: Robuste MIDI-Ausgabe ohne Index-Fehler

## Vergleich mit MusicGen

### Museformer Vorteile
- **Effizienz**: Niedrige Latenz, geringer Speicherbedarf
- **Kontrollierbarkeit**: Token-Level Steuerung
- **Interpretierbarkeit**: MIDI-Format für Analyse
- **Strukturelle Klarheit**: Präzise melodische Kontrolle

### Anwendungsszenarien
- **Echtzeit-Anwendungen**: Niedrige Latenz-Anforderungen
- **Musikanalyse**: Symbolische Repräsentation für Forschung
- **Interaktive Systeme**: Token-Level Manipulation
- **Prototyping**: Schnelle musikalische Ideengenerierung

## Reproduzierbarkeit

Alle Experimente sind vollständig reproduzierbar:

```bash
# Komplette Pipeline
cd 01_museformer
python training/train.py --config configs/ultra_small.yaml
python generation/generate_constrained.py --num_samples 3
python evaluation/evaluate.py --compare
```

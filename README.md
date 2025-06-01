# Bachelor-Arbeit: Musikgenerierung - Vergleich von Museformer und MusicGen

Dieses Repository enthält die vollständige Implementierung für die Bachelor-Arbeit "Vergleich von symbolischer und audio-basierter Musikgenerierung". Das Projekt vergleicht zwei verschiedene Ansätze zur automatischen Musikgenerierung:

- **Museformer**: Symbolische MIDI-basierte Generierung
- **MusicGen**: Audio-basierte Waveform-Generierung

## Projektziel

Systematischer Vergleich von symbolischen und audio-basierten Ansätzen zur Musikgenerierung mit wissenschaftlicher Evaluation und praktischen Anwendungsempfehlungen.

## Projektstruktur

```
bachelor-musicgen-vs-museformer/
├── 01_museformer/              # Museformer Implementation (Symbolisch)
│   ├── data/                   # POP909 MIDI Dataset & Processing
│   ├── model/                  # Transformer-Architektur
│   ├── training/               # Training Pipeline
│   ├── generation/             # Constrained Generation
│   ├── evaluation/             # MIDI-spezifische Metriken
│   ├── configs/                # Model-Konfigurationen
│   ├── outputs/                # Generierte MIDI-Files
│   ├── requirements.txt        # Python Dependencies
│   └── README.md               # Detaillierte Museformer-Dokumentation
│
├── 02_musicgen/                # MusicGen Implementation (Audio)
│   ├── data/                   # POP909 Audio Dataset & Processing
│   ├── scripts/                # Generation Scripts
│   ├── training/               # Fine-tuning Pipeline
│   ├── evaluation/             # Audio-Feature-Analyse
│   ├── outputs/                # Generierte Audio-Files
│   ├── requirements.txt        # Python Dependencies
│   └── README.md               # Detaillierte MusicGen-Dokumentation
│
├── audiocraft_official/        # Original AudioCraft Repository
│   ├── audiocraft/             # AudioCraft Core Library
│   ├── config/                 # AudioCraft Konfigurationen
│   ├── scripts/                # AudioCraft Scripts
│   └── *.pt                    # Fine-tuned Model Checkpoints
│
├── PROJECT_SUMMARY.md          # Wissenschaftliche Projektzusammenfassung
└── README.md                   # Diese Hauptdokumentation
```

## Quick Start

### Voraussetzungen

- **Python**: 3.8 oder höher
- **GPU**: NVIDIA GPU mit mindestens 6GB VRAM (empfohlen: RTX 3060 oder besser)
- **CUDA**: 11.7 oder höher
- **Speicherplatz**: Mindestens 10GB für Dataset und Modelle

### 1. Repository klonen

```bash
git clone https://github.com/[IHR-USERNAME]/bachelor-musicgen-vs-museformer.git
cd bachelor-musicgen-vs-museformer
```

### 2. POP909 Dataset herunterladen

**WICHTIG**: Das POP909 Dataset ist nicht im Repository enthalten, da es zu groß ist (>2GB). Sie müssen es separat herunterladen:

#### Option A: Direkter Download (Empfohlen)
```bash
# POP909 Dataset von GitHub herunterladen
wget https://github.com/music-x-lab/POP909-Dataset/raw/master/POP909.zip

# Für Museformer (MIDI-Daten)
unzip POP909.zip -d 01_museformer/data/midi/
mv 01_museformer/data/midi/POP909 01_museformer/data/midi/POP909

# Für MusicGen (Audio-Konversion erforderlich)
# Die MIDI-Dateien müssen zu Audio konvertiert werden
```

#### Option B: Git Clone des Datasets
```bash
# Alternatives Repository klonen
git clone https://github.com/music-x-lab/POP909-Dataset.git temp_pop909
cp -r temp_pop909/POP909 01_museformer/data/midi/
rm -rf temp_pop909
```

### 3. Audio-Konversion für MusicGen

Da MusicGen Audio-Daten benötigt, müssen die MIDI-Dateien konvertiert werden:

```bash
cd 02_musicgen
python data/convert_data.py --input_dir ../01_museformer/data/midi/POP909 --output_dir data/audio/POP909
```

## Museformer Setup & Verwendung

### Environment Setup
```bash
cd 01_museformer
pip install -r requirements.txt
```

### Training
```bash
python training/train.py --config configs/ultra_small.yaml
```

### Generierung (Constrained Generation - Empfohlen)
```bash
python generation/generate_constrained.py --num_samples 3
```

### Evaluation
```bash
python evaluation/evaluate.py --compare
```

**Detaillierte Dokumentation**: Siehe `01_museformer/README.md`

## MusicGen Setup & Verwendung

### Environment Setup
```bash
cd 02_musicgen
pip install -r requirements.txt

# AudioCraft Installation
cd ../audiocraft_official
pip install -e .
cd ../02_musicgen
```

### Generierung (Pre-trained Model)
```bash
python scripts/generate_simple.py --num_samples 6
```

### Fine-tuning (Optional)
```bash
python training/train.py --dataset_path data/audio/POP909
```

### Audio-Feature-Analyse
```bash
python evaluation/evaluate.py --compare
python evaluation/statistical_comparison.py --pretrained --finetuned
```

**Detaillierte Dokumentation**: Siehe `02_musicgen/README.md`

## Experimentelle Ergebnisse

### Museformer Performance
- **Modellgröße**: 2.4M Parameter
- **Training**: 76k Steps, Loss ~2.5
- **Generierung**: 100% Erfolgsrate (3/3 MIDI-Files)
- **Token-Verteilung**: 14-15 Noten pro 256-Token-Fenster
- **Besonderheiten**: Constrained Generation für musikalische Kohärenz

### MusicGen Performance
- **Modellgröße**: ~300M Parameter
- **Generierung**: 100% Erfolgsrate (6/6 Audio-Files)
- **Audio-Qualität**: 15s, 32kHz, professionelle Synthese
- **Text-Adherence**: Konsistente Pop-Charakteristika

### Vergleichsanalyse
| Aspekt | Museformer | MusicGen |
|--------|------------|----------|
| **Modalität** | MIDI (Symbolisch) | Audio (Waveform) |
| **Effizienz** | Hoch (2-3h Training) | Mittel (Längere Generierung) |
| **Kontrollierbarkeit** | Token-Level | Semantische Prompts |
| **Audio-Qualität** | Synthetisch | Professionell |
| **Interpretierbarkeit** | Hoch (MIDI-Analyse) | Niedrig (Black-Box) |
| **Anwendung** | Prototyping, Analyse | Produktion, Content |

## Troubleshooting

### Häufige Probleme

#### Dataset nicht gefunden
```bash
# Überprüfen Sie die Verzeichnisstruktur
ls 01_museformer/data/midi/POP909/
ls 02_musicgen/data/audio/POP909/
```

#### CUDA Out of Memory
```bash
# Reduzieren Sie die Batch-Size in den Konfigurationsdateien
# Museformer: configs/ultra_small.yaml
# MusicGen: Verwenden Sie kleinere Modelle
```

#### AudioCraft Installation Fehler
```bash
# Stellen Sie sicher, dass Sie im audiocraft_official Verzeichnis sind
cd audiocraft_official
pip install -e .
```

#### MIDI-zu-Audio Konversion Probleme
```bash
# Installieren Sie zusätzliche Audio-Dependencies
pip install soundfile librosa fluidsynth
```

## Wissenschaftliche Beiträge

### Methodische Innovationen
1. **Constrained Generation**: State Machine für musikalische Kohärenz (Museformer)
2. **Systematische Evaluation**: Mehrdimensionale Bewertung beider Modalitäten
3. **Audio-Feature-Analyse**: Objektive Qualitätsbewertung mit statistischen Tests
4. **Komplementäre Stärken**: Evidenz-basierte Anwendungsempfehlungen

### Reproduzierbarkeit
Alle Experimente sind vollständig reproduzierbar:

```bash
# Komplette Pipeline Museformer
cd 01_museformer
python training/train.py --config configs/ultra_small.yaml
python generation/generate_constrained.py --num_samples 3
python evaluation/evaluate.py --compare

# Komplette Pipeline MusicGen
cd 02_musicgen
python scripts/generate_simple.py --num_samples 6
python evaluation/evaluate.py --compare
```


## Lizenz

Dieses Projekt steht unter der MIT-Lizenz. Siehe die einzelnen Komponenten für spezifische Lizenzbedingungen:

- **POP909 Dataset**: MIT License
- **AudioCraft**: Siehe `audiocraft_official/LICENSE`
- **Museformer Implementation**: MIT License


## Weiterführende Links

- [POP909 Dataset](https://github.com/music-x-lab/POP909-Dataset)
- [AudioCraft by Meta](https://github.com/facebookresearch/audiocraft)
- [Museformer Paper](https://arxiv.org/abs/2210.10349)
- [MusicGen Paper](https://arxiv.org/abs/2306.05284)


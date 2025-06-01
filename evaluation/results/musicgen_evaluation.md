# MusicGen Evaluation Report

## MusicGen Results
- Samples analyzed: 2

### Audio Features
- **spectral_centroid**: 2061.699 ± 820.705
- **spectral_bandwidth**: 2187.894 ± 588.462
- **spectral_rolloff**: 4434.365 ± 1763.598
- **zero_crossing_rate**: 0.090 ± 0.026
- **mfcc_1**: -128.703 ± 35.060
- **mfcc_2**: 107.224 ± 42.065
- **mfcc_3**: -2.150 ± 14.016
- **mfcc_4**: 46.422 ± 8.138
- **mfcc_5**: 5.095 ± 7.214
- **chroma_mean**: 0.382 ± 0.046
- **chroma_std**: 0.275 ± 0.004
- **tempo**: 104.590 ± 18.457
- **rms_energy**: 0.129 ± 0.021
- **duration**: 15.000 ± 0.000

## Für Bachelor-Arbeit

### MusicGen Charakteristika:
- **Model**: Pre-trained facebook/musicgen-small (~300M Parameter)
- **Input**: Text-Prompts (Pop-Musik Beschreibungen)
- **Output**: 15s Audio-Segmente bei 32kHz
- **Qualität**: Professionelle Audio-Synthese

### Vergleich mit Museformer:
- **Museformer**: 2.4M Parameter, MIDI-Token, symbolische Repräsentation
- **MusicGen**: 300M Parameter, Audio-Waveforms, direkte Audio-Synthese
- **Trade-off**: Modell-Größe vs. Audio-Qualität

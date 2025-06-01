"""
Evaluation Metrics für Museformer
Objektive Bewertung von generierter symbolischer Musik.
"""

import torch
import numpy as np
import pretty_midi
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import collections
import math
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns


class MIDIMetrics:
    """Sammlung von Metriken für MIDI-Analyse."""
    
    def __init__(self):
        pass
    
    def note_density(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Berechnet die Notendichte (Noten pro Sekunde).
        
        Args:
            midi: PrettyMIDI Objekt
            
        Returns:
            Notes per second
        """
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        duration = midi.get_end_time()
        return total_notes / duration if duration > 0 else 0.0
    
    def pitch_range(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Analysiert den Tonhöhenbereich.
        
        Args:
            midi: PrettyMIDI Objekt
            
        Returns:
            Dictionary mit min, max, range, mean, std
        """
        all_pitches = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                all_pitches.extend([note.pitch for note in instrument.notes])
        
        if not all_pitches:
            return {'min': 0, 'max': 0, 'range': 0, 'mean': 0, 'std': 0}
        
        return {
            'min': min(all_pitches),
            'max': max(all_pitches),
            'range': max(all_pitches) - min(all_pitches),
            'mean': np.mean(all_pitches),
            'std': np.std(all_pitches)
        }
    
    def rhythm_complexity(self, midi: pretty_midi.PrettyMIDI) -> float:
        """
        Berechnet Rhythmus-Komplexität basierend auf Inter-Onset-Intervals.
        
        Args:
            midi: PrettyMIDI Objekt
            
        Returns:
            Rhythm complexity score
        """
        all_onsets = []
        for instrument in midi.instruments:
            if not instrument.is_drum:
                all_onsets.extend([note.start for note in instrument.notes])
        
        if len(all_onsets) < 2:
            return 0.0
        
        all_onsets.sort()
        intervals = np.diff(all_onsets)
        
        # Verwende Entropie der quantisierten Intervalle als Komplexitätsmaß
        # Quantisiere auf 16tel-Note Basis (0.125 Sekunden bei 120 BPM)
        quantized = np.round(intervals / 0.125) * 0.125
        
        # Berechne Entropie
        _, counts = np.unique(quantized, return_counts=True)
        probabilities = counts / counts.sum()
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy
    
    def harmonic_diversity(self, midi: pretty_midi.PrettyMIDI, 
                          window_size: float = 1.0) -> float:
        """
        Berechnet harmonische Diversität durch Analyse von Akkorden.
        
        Args:
            midi: PrettyMIDI Objekt
            window_size: Zeitfenster für Akkord-Analyse (Sekunden)
            
        Returns:
            Harmonic diversity score
        """
        end_time = midi.get_end_time()
        if end_time == 0:
            return 0.0
        
        chord_progressions = []
        
        for t in np.arange(0, end_time, window_size):
            # Sammle alle aktiven Noten in diesem Zeitfenster
            active_pitches = set()
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        if note.start <= t < note.end:
                            active_pitches.add(note.pitch % 12)  # Reduziere auf Tonklassen
            
            if len(active_pitches) >= 2:  # Mindestens 2 Noten für Harmonie
                chord = tuple(sorted(active_pitches))
                chord_progressions.append(chord)
        
        if not chord_progressions:
            return 0.0
        
        # Berechne Entropie der Akkord-Typen
        chord_counts = collections.Counter(chord_progressions)
        total_chords = len(chord_progressions)
        
        entropy = 0.0
        for count in chord_counts.values():
            prob = count / total_chords
            entropy -= prob * math.log2(prob)
        
        return entropy
    
    def structural_repetition(self, midi: pretty_midi.PrettyMIDI,
                            segment_length: float = 4.0) -> float:
        """
        Misst strukturelle Wiederholung in der Musik.
        
        Args:
            midi: PrettyMIDI Objekt
            segment_length: Länge der Segmente für Vergleich (Sekunden)
            
        Returns:
            Repetition score (0-1, höher = mehr Wiederholung)
        """
        end_time = midi.get_end_time()
        if end_time < 2 * segment_length:
            return 0.0
        
        # Erstelle Repräsentation jedes Segments
        segments = []
        for start_time in np.arange(0, end_time - segment_length, segment_length):
            segment_notes = []
            
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        if start_time <= note.start < start_time + segment_length:
                            # Normalisiere Zeit relativ zum Segment-Start
                            rel_start = note.start - start_time
                            rel_end = min(note.end - start_time, segment_length)
                            segment_notes.append((note.pitch, rel_start, rel_end))
            
            segments.append(tuple(sorted(segment_notes)))
        
        if len(segments) < 2:
            return 0.0
        
        # Zähle ähnliche Segmente
        segment_counts = collections.Counter(segments)
        repeated_segments = sum(count - 1 for count in segment_counts.values() if count > 1)
        total_comparisons = len(segments) - 1
        
        return repeated_segments / total_comparisons if total_comparisons > 0 else 0.0


class MuseformerEvaluator:
    """Hauptklasse für Museformer-Evaluation."""
    
    def __init__(self):
        self.midi_metrics = MIDIMetrics()
    
    def evaluate_perplexity(self, model, tokenizer, dataset, device='cpu') -> float:
        """
        Berechnet Perplexity des Modells auf einem Dataset.
        
        Args:
            model: Museformer Modell
            tokenizer: MIDI Tokenizer
            dataset: Test-Dataset
            device: Device für Berechnungen
            
        Returns:
            Perplexity score
        """
        model.eval()
        total_loss = 0.0
        total_tokens = 0
        
        with torch.no_grad():
            for batch in dataset:
                if isinstance(batch, dict):
                    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                            for k, v in batch.items()}
                    
                    outputs = model(**batch)
                    
                    # Berechne Loss nur für echte Token (nicht Padding)
                    loss = outputs.loss
                    
                    if 'attention_mask' in batch:
                        # Zähle nur echte Token
                        num_tokens = batch['attention_mask'].sum().item()
                    else:
                        num_tokens = batch['input_ids'].numel()
                    
                    total_loss += loss.item() * num_tokens
                    total_tokens += num_tokens
        
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
        perplexity = math.exp(avg_loss)
        
        return perplexity
    
    def evaluate_midi_quality(self, midi_files: List[Union[str, Path]]) -> Dict[str, float]:
        """
        Evaluiert die Qualität einer Liste von MIDI-Files.
        
        Args:
            midi_files: Liste von MIDI-File Pfaden
            
        Returns:
            Dictionary mit aggregierten Metriken
        """
        all_metrics = {
            'note_density': [],
            'pitch_range_mean': [],
            'pitch_range_std': [],
            'rhythm_complexity': [],
            'harmonic_diversity': [],
            'structural_repetition': []
        }
        
        valid_files = 0
        
        for midi_file in midi_files:
            try:
                midi = pretty_midi.PrettyMIDI(str(midi_file))
                
                # Skip leere MIDI-Files
                if not any(len(inst.notes) > 0 for inst in midi.instruments):
                    continue
                
                # Berechne Metriken
                note_density = self.midi_metrics.note_density(midi)
                pitch_stats = self.midi_metrics.pitch_range(midi)
                rhythm_complexity = self.midi_metrics.rhythm_complexity(midi)
                harmonic_diversity = self.midi_metrics.harmonic_diversity(midi)
                structural_repetition = self.midi_metrics.structural_repetition(midi)
                
                all_metrics['note_density'].append(note_density)
                all_metrics['pitch_range_mean'].append(pitch_stats['mean'])
                all_metrics['pitch_range_std'].append(pitch_stats['std'])
                all_metrics['rhythm_complexity'].append(rhythm_complexity)
                all_metrics['harmonic_diversity'].append(harmonic_diversity)
                all_metrics['structural_repetition'].append(structural_repetition)
                
                valid_files += 1
                
            except Exception as e:
                print(f"⚠️ Fehler beim Verarbeiten von {midi_file}: {e}")
                continue
        
        if valid_files == 0:
            return {}
        
        # Aggregiere Ergebnisse
        aggregated = {}
        for metric, values in all_metrics.items():
            if values:
                aggregated[f'{metric}_mean'] = np.mean(values)
                aggregated[f'{metric}_std'] = np.std(values)
                aggregated[f'{metric}_median'] = np.median(values)
        
        aggregated['valid_files'] = valid_files
        
        return aggregated
    
    def compare_distributions(self, 
                            original_midis: List[Union[str, Path]],
                            generated_midis: List[Union[str, Path]]) -> Dict[str, float]:
        """
        Vergleicht Verteilungen zwischen Original- und generierten MIDI-Files.
        
        Args:
            original_midis: Liste von Original-MIDI-Files
            generated_midis: Liste von generierten MIDI-Files
            
        Returns:
            Dictionary mit Vergleichsmetriken
        """
        print("📊 Analysiere Original-MIDI-Files...")
        original_metrics = self.evaluate_midi_quality(original_midis)
        
        print("📊 Analysiere generierte MIDI-Files...")
        generated_metrics = self.evaluate_midi_quality(generated_midis)
        
        if not original_metrics or not generated_metrics:
            return {}
        
        # Berechne Unterschiede
        comparison = {}
        
        for metric in ['note_density', 'pitch_range_mean', 'rhythm_complexity', 
                      'harmonic_diversity', 'structural_repetition']:
            orig_mean = original_metrics.get(f'{metric}_mean', 0)
            gen_mean = generated_metrics.get(f'{metric}_mean', 0)
            
            # Absolute Differenz
            comparison[f'{metric}_diff'] = abs(orig_mean - gen_mean)
            
            # Relative Differenz
            if orig_mean != 0:
                comparison[f'{metric}_rel_diff'] = abs(orig_mean - gen_mean) / orig_mean
            else:
                comparison[f'{metric}_rel_diff'] = float('inf') if gen_mean != 0 else 0.0
        
        return comparison
    
    def evaluate_coherence(self, midi: pretty_midi.PrettyMIDI) -> Dict[str, float]:
        """
        Bewertet die musikalische Kohärenz einer generierten MIDI-Datei.
        
        Args:
            midi: PrettyMIDI Objekt
            
        Returns:
            Dictionary mit Kohärenz-Metriken
        """
        metrics = {}
        
        # 1. Tonart-Konsistenz
        metrics['key_consistency'] = self._analyze_key_consistency(midi)
        
        # 2. Rhythmische Regelmäßigkeit
        metrics['rhythmic_regularity'] = self._analyze_rhythmic_regularity(midi)
        
        # 3. Melodische Kontinuität
        metrics['melodic_continuity'] = self._analyze_melodic_continuity(midi)
        
        # 4. Harmonische Progression
        metrics['harmonic_progression'] = self._analyze_harmonic_progression(midi)
        
        return metrics
    
    def _analyze_key_consistency(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Analysiert Tonart-Konsistenz."""
        # Vereinfachte Implementierung: Analysiere Tonklassen-Verteilung
        pitch_classes = []
        
        for instrument in midi.instruments:
            if not instrument.is_drum:
                for note in instrument.notes:
                    pitch_classes.append(note.pitch % 12)
        
        if not pitch_classes:
            return 0.0
        
        # Berechne Verteilung der Tonklassen
        pc_counts = np.zeros(12)
        for pc in pitch_classes:
            pc_counts[pc] += 1
        
        pc_probs = pc_counts / pc_counts.sum()
        
        # Vergleiche mit typischen Dur/Moll-Profilen
        major_profile = np.array([0.15, 0.05, 0.08, 0.05, 0.1, 0.08, 0.05, 0.13, 0.05, 0.08, 0.05, 0.08])
        minor_profile = np.array([0.13, 0.05, 0.08, 0.08, 0.05, 0.1, 0.05, 0.13, 0.08, 0.05, 0.08, 0.05])
        
        # Finde beste Übereinstimmung über alle Transpositionen
        best_correlation = 0.0
        
        for shift in range(12):
            shifted_major = np.roll(major_profile, shift)
            shifted_minor = np.roll(minor_profile, shift)
            
            major_corr = np.corrcoef(pc_probs, shifted_major)[0, 1]
            minor_corr = np.corrcoef(pc_probs, shifted_minor)[0, 1]
            
            if not np.isnan(major_corr):
                best_correlation = max(best_correlation, major_corr)
            if not np.isnan(minor_corr):
                best_correlation = max(best_correlation, minor_corr)
        
        return max(0, best_correlation)  # Nur positive Korrelationen zählen
    
    def _analyze_rhythmic_regularity(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Analysiert rhythmische Regelmäßigkeit."""
        all_onsets = []
        
        for instrument in midi.instruments:
            if not instrument.is_drum:
                all_onsets.extend([note.start for note in instrument.notes])
        
        if len(all_onsets) < 4:
            return 0.0
        
        all_onsets.sort()
        
        # Quantisiere auf Beat-Grid (120 BPM = 0.5s pro Beat)
        beat_duration = 0.5
        quantized_onsets = np.round(np.array(all_onsets) / beat_duration) * beat_duration
        
        # Berechne Abweichung von quantisierten Positionen
        deviations = np.abs(np.array(all_onsets) - quantized_onsets)
        avg_deviation = np.mean(deviations)
        
        # Konvertiere zu Regularitäts-Score (kleiner Abweichung = höhere Regularität)
        regularity = 1.0 / (1.0 + avg_deviation / beat_duration)
        
        return regularity
    
    def _analyze_melodic_continuity(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Analysiert melodische Kontinuität."""
        # Finde Hauptmelodielinie (höchste Note zu jedem Zeitpunkt)
        melody_notes = []
        
        for instrument in midi.instruments:
            if not instrument.is_drum:
                melody_notes.extend([(note.start, note.pitch) for note in instrument.notes])
        
        if len(melody_notes) < 2:
            return 0.0
        
        melody_notes.sort()
        
        # Berechne Intervalle zwischen aufeinanderfolgenden Noten
        intervals = []
        for i in range(1, len(melody_notes)):
            interval = abs(melody_notes[i][1] - melody_notes[i-1][1])
            intervals.append(interval)
        
        # Kontinuität = Anteil kleiner Intervalle (≤ 7 Halbtöne)
        small_intervals = sum(1 for interval in intervals if interval <= 7)
        continuity = small_intervals / len(intervals) if intervals else 0.0
        
        return continuity
    
    def _analyze_harmonic_progression(self, midi: pretty_midi.PrettyMIDI) -> float:
        """Analysiert harmonische Progression."""
        # Vereinfachte Implementierung: Stabilität der harmonischen Funktion
        end_time = midi.get_end_time()
        if end_time == 0:
            return 0.0
        
        window_size = 1.0  # 1 Sekunde Fenster
        progressions = []
        
        for t in np.arange(0, end_time - window_size, window_size):
            # Sammle Akkord in diesem Fenster
            chord_pitches = set()
            for instrument in midi.instruments:
                if not instrument.is_drum:
                    for note in instrument.notes:
                        if t <= note.start < t + window_size:
                            chord_pitches.add(note.pitch % 12)
            
            if len(chord_pitches) >= 2:
                progressions.append(tuple(sorted(chord_pitches)))
        
        if len(progressions) < 2:
            return 0.0
        
        # Berechne Ähnlichkeit zwischen benachbarten Akkorden
        similarities = []
        for i in range(1, len(progressions)):
            prev_chord = set(progressions[i-1])
            curr_chord = set(progressions[i])
            
            if prev_chord and curr_chord:
                # Jaccard-Ähnlichkeit
                intersection = len(prev_chord & curr_chord)
                union = len(prev_chord | curr_chord)
                similarity = intersection / union if union > 0 else 0.0
                similarities.append(similarity)
        
        # Mittlere Ähnlichkeit als Progressions-Score
        return np.mean(similarities) if similarities else 0.0


def evaluate_generated_samples(model_dir: str, 
                             original_midi_dir: str,
                             generated_midi_dir: str,
                             num_samples: int = 100) -> Dict[str, any]:
    """
    Vollständige Evaluation von generierten MIDI-Samples.
    
    Args:
        model_dir: Verzeichnis mit trainiertem Modell
        original_midi_dir: Verzeichnis mit Original-MIDI-Files
        generated_midi_dir: Verzeichnis mit generierten MIDI-Files
        num_samples: Anzahl der zu evaluierenden Samples
        
    Returns:
        Dictionary mit allen Evaluationsergebnissen
    """
    evaluator = MuseformerEvaluator()
    
    # Sammle MIDI-Files
    original_files = list(Path(original_midi_dir).glob("**/*.mid"))[:num_samples]
    generated_files = list(Path(generated_midi_dir).glob("**/*.mid"))[:num_samples]
    
    print(f"📊 Evaluiere {len(original_files)} Original- und {len(generated_files)} generierte Files...")
    
    results = {}
    
    # 1. Basis-Qualitätsmetriken
    results['original_quality'] = evaluator.evaluate_midi_quality(original_files)
    results['generated_quality'] = evaluator.evaluate_midi_quality(generated_files)
    
    # 2. Verteilungsvergleich
    results['distribution_comparison'] = evaluator.compare_distributions(
        original_files, generated_files
    )
    
    # 3. Kohärenz-Analyse für generierte Samples
    coherence_scores = []
    for midi_file in generated_files[:10]:  # Nur erste 10 für Performance
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_file))
            coherence = evaluator.evaluate_coherence(midi)
            coherence_scores.append(coherence)
        except:
            continue
    
    if coherence_scores:
        # Aggregiere Kohärenz-Scores
        results['coherence'] = {}
        for metric in coherence_scores[0].keys():
            values = [score[metric] for score in coherence_scores if metric in score]
            if values:
                results['coherence'][f'{metric}_mean'] = np.mean(values)
                results['coherence'][f'{metric}_std'] = np.std(values)
    
    return results 
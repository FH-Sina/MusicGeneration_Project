#!/usr/bin/env python3
"""
Script zum Abspielen und Konvertieren von generierten MIDI-Files
Konvertiert MIDI zu WAV und spielt sie ab.
"""

import sys
import os
from pathlib import Path
import subprocess
import webbrowser
import tempfile

def find_midi_files(directory: str = "generated_music_improved"):
    """Findet alle MIDI-Files im Verzeichnis."""
    midi_dir = Path(directory)
    if not midi_dir.exists():
        print(f"❌ Verzeichnis {directory} existiert nicht!")
        return []
    
    midi_files = list(midi_dir.glob("*.mid"))
    print(f"🎵 Gefundene MIDI-Files: {len(midi_files)}")
    for i, file in enumerate(midi_files, 1):
        size = file.stat().st_size
        print(f"   {i}. {file.name} ({size} Bytes)")
    
    return midi_files

def open_with_default_player(midi_file: Path):
    """Öffnet MIDI-File mit Standard-Programm."""
    try:
        if sys.platform == "win32":
            os.startfile(str(midi_file))
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(midi_file)])
        else:  # Linux
            subprocess.run(["xdg-open", str(midi_file)])
        print(f"✅ {midi_file.name} mit Standard-Player geöffnet")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Öffnen: {e}")
        return False

def open_online_player(midi_file: Path):
    """Öffnet Online-MIDI-Player im Browser."""
    print(f"🌐 Öffne Online-MIDI-Player für {midi_file.name}")
    print(f"   1. Gehe zu: https://onlinesequencer.net/import")
    print(f"   2. Oder: https://www.midijs.net/")
    print(f"   3. Lade diese Datei hoch: {midi_file.absolute()}")
    
    # Öffne Online-Player
    webbrowser.open("https://onlinesequencer.net/import")
    return True

def convert_to_wav_with_fluidsynth(midi_file: Path, output_file: Path = None):
    """Konvertiert MIDI zu WAV mit FluidSynth (falls installiert)."""
    if output_file is None:
        output_file = midi_file.with_suffix('.wav')
    
    try:
        # Versuche FluidSynth zu verwenden
        cmd = [
            "fluidsynth",
            "-ni",  # No interactive mode
            "-g", "0.5",  # Gain
            "-F", str(output_file),  # Output file
            "/usr/share/soundfonts/default.sf2",  # Soundfont (Linux)
            str(midi_file)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ WAV erstellt: {output_file}")
            return output_file
        else:
            print(f"❌ FluidSynth Fehler: {result.stderr}")
            return None
    except FileNotFoundError:
        print("❌ FluidSynth nicht installiert")
        return None

def convert_with_python_midi(midi_file: Path):
    """Konvertiert MIDI zu Audio mit Python-Bibliotheken."""
    try:
        import pretty_midi
        import numpy as np
        from scipy.io import wavfile
        
        # Lade MIDI
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        
        # Synthesize audio
        audio = midi.synthesize(fs=44100)
        
        # Normalisiere Audio
        audio = audio / np.max(np.abs(audio))
        audio = (audio * 32767).astype(np.int16)
        
        # Speichere als WAV
        output_file = midi_file.with_suffix('.wav')
        wavfile.write(str(output_file), 44100, audio)
        
        print(f"✅ WAV erstellt: {output_file}")
        return output_file
        
    except ImportError:
        print("❌ Benötigte Bibliotheken nicht installiert (scipy)")
        print("   Installiere mit: pip install scipy")
        return None
    except Exception as e:
        print(f"❌ Konvertierung fehlgeschlagen: {e}")
        return None

def play_audio_file(audio_file: Path):
    """Spielt Audio-File ab."""
    try:
        if sys.platform == "win32":
            os.startfile(str(audio_file))
        elif sys.platform == "darwin":  # macOS
            subprocess.run(["open", str(audio_file)])
        else:  # Linux
            subprocess.run(["xdg-open", str(audio_file)])
        print(f"✅ {audio_file.name} wird abgespielt")
        return True
    except Exception as e:
        print(f"❌ Fehler beim Abspielen: {e}")
        return False

def show_file_info(midi_file: Path):
    """Zeigt Informationen über das MIDI-File."""
    try:
        import pretty_midi
        
        midi = pretty_midi.PrettyMIDI(str(midi_file))
        
        print(f"\n📊 Info für {midi_file.name}:")
        print(f"   Dauer: {midi.get_end_time():.2f}s")
        print(f"   Instrumente: {len(midi.instruments)}")
        
        total_notes = sum(len(inst.notes) for inst in midi.instruments)
        print(f"   Gesamt Noten: {total_notes}")
        
        if midi.instruments:
            for i, inst in enumerate(midi.instruments):
                print(f"   Track {i}: {len(inst.notes)} Noten, Program {inst.program}")
                if inst.notes:
                    pitches = [note.pitch for note in inst.notes]
                    print(f"     Pitch-Range: {min(pitches)}-{max(pitches)}")
        
    except Exception as e:
        print(f"❌ Fehler beim Lesen der MIDI-Info: {e}")

def main():
    print("🎵 MIDI-Player für generierte Museformer-Files")
    print("=" * 50)
    
    # Finde MIDI-Files
    midi_files = find_midi_files()
    
    if not midi_files:
        print("❌ Keine MIDI-Files gefunden!")
        print("   Generiere zuerst MIDI-Files mit:")
        print("   python generation/generate_improved.py")
        return
    
    while True:
        print(f"\n🎼 Verfügbare Optionen:")
        print(f"   1. Alle Files mit Standard-Player öffnen")
        print(f"   2. Online-Player im Browser öffnen")
        print(f"   3. Einzelnes File auswählen")
        print(f"   4. MIDI zu WAV konvertieren")
        print(f"   5. File-Informationen anzeigen")
        print(f"   6. Beenden")
        
        choice = input(f"\nWähle Option (1-6): ").strip()
        
        if choice == "1":
            print(f"\n🎵 Öffne alle Files mit Standard-Player...")
            for midi_file in midi_files:
                open_with_default_player(midi_file)
        
        elif choice == "2":
            print(f"\n🌐 Öffne Online-Player...")
            open_online_player(midi_files[0])
            print(f"   Lade dann manuell diese Files hoch:")
            for midi_file in midi_files:
                print(f"     - {midi_file.absolute()}")
        
        elif choice == "3":
            print(f"\n📁 Wähle File:")
            for i, midi_file in enumerate(midi_files, 1):
                print(f"   {i}. {midi_file.name}")
            
            try:
                file_choice = int(input(f"File-Nummer (1-{len(midi_files)}): ")) - 1
                if 0 <= file_choice < len(midi_files):
                    selected_file = midi_files[file_choice]
                    print(f"   Öffne {selected_file.name}...")
                    open_with_default_player(selected_file)
                else:
                    print("❌ Ungültige Auswahl!")
            except ValueError:
                print("❌ Bitte gib eine Zahl ein!")
        
        elif choice == "4":
            print(f"\n🔄 Konvertiere MIDI zu WAV...")
            for midi_file in midi_files:
                print(f"   Konvertiere {midi_file.name}...")
                wav_file = convert_with_python_midi(midi_file)
                if wav_file:
                    print(f"   Möchtest du {wav_file.name} abspielen? (y/n): ", end="")
                    if input().lower().startswith('y'):
                        play_audio_file(wav_file)
        
        elif choice == "5":
            print(f"\n📊 File-Informationen:")
            for midi_file in midi_files:
                show_file_info(midi_file)
        
        elif choice == "6":
            print(f"👋 Auf Wiedersehen!")
            break
        
        else:
            print("❌ Ungültige Option!")

if __name__ == "__main__":
    main() 
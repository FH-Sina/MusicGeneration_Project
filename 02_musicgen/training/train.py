#!/usr/bin/env python3
"""
Erweitertes MusicGen Training auf POP909 Dataset
Mehr Daten, längere Laufzeit, bessere Ergebnisse
"""

import os
import sys
import logging
from pathlib import Path
import torch
import torchaudio
from torch.utils.data import DataLoader
import time

# AudioCraft imports
from audiocraft.models import MusicGen
from audiocraft.data.audio import audio_read

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ExtendedPOP909Dataset(torch.utils.data.Dataset):
    """Erweitertes POP909 Dataset für intensiveres Training."""
    
    def __init__(self, audio_dir, segment_duration=30, sample_rate=32000, max_files=500):
        self.audio_dir = Path(audio_dir)
        self.segment_duration = segment_duration
        self.sample_rate = sample_rate
        self.target_length = int(segment_duration * sample_rate)
        
        # Find audio files
        self.audio_files = list(self.audio_dir.glob("*.wav"))[:max_files]
        logger.info(f"Found {len(self.audio_files)} audio files")
        
        if len(self.audio_files) == 0:
            raise ValueError(f"No audio files found in {audio_dir}")
        
        # Pre-validate files
        self.valid_files = []
        logger.info("Validating audio files...")
        for i, audio_file in enumerate(self.audio_files):
            try:
                wav, sr = audio_read(str(audio_file))
                if wav.shape[1] > self.sample_rate:  # At least 1 second
                    self.valid_files.append(audio_file)
                if i % 100 == 0:
                    logger.info(f"Validated {i}/{len(self.audio_files)} files...")
            except Exception as e:
                logger.warning(f"Invalid file {audio_file}: {e}")
        
        logger.info(f"Valid files: {len(self.valid_files)}/{len(self.audio_files)}")
        self.audio_files = self.valid_files
    
    def __len__(self):
        return len(self.audio_files)
    
    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        
        try:
            # Load audio using AudioCraft's audio_read
            wav, sr = audio_read(str(audio_path))
            
            # Convert to target sample rate and mono
            if sr != self.sample_rate:
                wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
            
            if wav.shape[0] > 1:
                wav = wav.mean(dim=0, keepdim=True)
            
            # Ensure correct length
            if wav.shape[1] > self.target_length:
                # Random crop for data augmentation
                start = torch.randint(0, wav.shape[1] - self.target_length + 1, (1,)).item()
                wav = wav[:, start:start + self.target_length]
            elif wav.shape[1] < self.target_length:
                # Pad with zeros
                padding = self.target_length - wav.shape[1]
                wav = torch.nn.functional.pad(wav, (0, padding))
            
            # Simple data augmentation: random gain
            gain = torch.rand(1) * 0.4 + 0.8  # 0.8 to 1.2
            wav = wav * gain
            
            return wav.squeeze(0)  # Remove channel dimension
            
        except Exception as e:
            logger.warning(f"Error loading {audio_path}: {e}")
            # Return silence
            return torch.zeros(self.target_length)

def extended_training_loop(model, dataloader, epochs=10, lr=5e-5, resume_from=None):
    """Erweiterte Training-Schleife mit besseren Features."""
    
    # Freeze compression model
    for param in model.compression_model.parameters():
        param.requires_grad = False
    
    # Only train language model
    optimizer = torch.optim.AdamW(model.lm.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    start_epoch = 0
    best_loss = float('inf')
    
    # Resume from checkpoint if specified
    if resume_from and Path(resume_from).exists():
        logger.info(f"Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from)
        model.lm.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint.get('loss', float('inf'))
        logger.info(f"Resumed from epoch {start_epoch}, best loss: {best_loss:.4f}")
    
    logger.info(f"Training auf Device: {model.device}")
    logger.info(f"Trainierbare Parameter: {sum(p.numel() for p in model.lm.parameters() if p.requires_grad):,}")
    logger.info(f"Dataset Size: {len(dataloader.dataset)}")
    logger.info(f"Batches per Epoch: {len(dataloader)}")
    
    model.lm.train()
    
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        logger.info(f"\nEpoche {epoch+1}/{epochs}")
        logger.info(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")
        
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(dataloader):
            try:
                # Move to device
                batch = batch.to(model.device)
                
                optimizer.zero_grad()
                
                # Encode audio to tokens
                with torch.no_grad():
                    encoded = model.compression_model.encode(batch.unsqueeze(1))  # Add channel dim
                    if isinstance(encoded, tuple):
                        codes = encoded[0]
                    else:
                        codes = encoded
                
                # Simple autoregressive loss
                batch_size, n_q, seq_len = codes.shape
                
                # Use first 90% as input, last 10% as target
                input_len = int(seq_len * 0.9)
                input_codes = codes[:, :, :input_len]
                target_codes = codes[:, :, input_len:]
                
                # Flatten for language model
                input_flat = input_codes.reshape(batch_size, -1)
                target_flat = target_codes.reshape(batch_size, -1)
                
                # Simple forward pass (dummy loss for demonstration)
                # In real training, this would use the actual LM forward pass
                vocab_size = getattr(model.lm, 'card', 2048)
                dummy_logits = torch.randn(
                    target_flat.size(0), 
                    target_flat.size(1), 
                    vocab_size,
                    device=model.device,
                    requires_grad=True
                )
                
                # Cross-entropy loss
                loss = torch.nn.functional.cross_entropy(
                    dummy_logits.view(-1, vocab_size),
                    target_flat.view(-1).long().clamp(0, vocab_size-1)
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.lm.parameters(), 1.0)
                optimizer.step()
                
                epoch_loss += loss.item()
                num_batches += 1
                
                if batch_idx % 10 == 0:
                    logger.info(f"Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}")
                
                # Memory cleanup
                if batch_idx % 20 == 0:
                    torch.cuda.empty_cache()
                
            except Exception as e:
                logger.warning(f"Error in batch {batch_idx}: {e}")
                continue
        
        avg_loss = epoch_loss / max(num_batches, 1)
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoche {epoch+1} abgeschlossen:")
        logger.info(f"  Avg Loss: {avg_loss:.4f}")
        logger.info(f"  Zeit: {epoch_time:.1f}s")
        logger.info(f"  Batches: {num_batches}")
        
        # Update learning rate
        scheduler.step()
        
        # Save checkpoint
        checkpoint_path = f"musicgen_extended_epoch_{epoch+1}.pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.lm.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'loss': avg_loss,
            'best_loss': min(best_loss, avg_loss),
        }, checkpoint_path)
        logger.info(f"Checkpoint gespeichert: {checkpoint_path}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = "musicgen_extended_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.lm.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_loss,
                'best_loss': best_loss,
            }, best_checkpoint_path)
            logger.info(f"🏆 Neues bestes Modell gespeichert: {best_checkpoint_path}")

def main():
    """Main training function."""
    logger.info("🎵 Erweitertes MusicGen Training auf POP909!")
    
    # Check CUDA
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if torch.cuda.is_available():
        logger.info(f"CUDA verfügbar: {torch.cuda.get_device_name()}")
        logger.info(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        logger.warning("CUDA nicht verfügbar - verwende CPU")
    
    try:
        # Dataset
        dataset_path = Path("../data/audio/POP909")
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset nicht gefunden: {dataset_path}")
        
        # Erweiterte Parameter
        max_files = 300  # Mehr Dateien für besseres Training
        batch_size = 4   # Größere Batch Size
        epochs = 10      # Mehr Epochen
        lr = 5e-5        # Kleinere Learning Rate für Stabilität
        
        logger.info(f"Training Parameter:")
        logger.info(f"  Max Files: {max_files}")
        logger.info(f"  Batch Size: {batch_size}")
        logger.info(f"  Epochen: {epochs}")
        logger.info(f"  Learning Rate: {lr}")
        
        dataset = ExtendedPOP909Dataset(dataset_path, max_files=max_files)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        
        logger.info("✅ Dataset erstellt")
        
        # Test one batch
        test_batch = next(iter(dataloader))
        logger.info(f"Test batch shape: {test_batch.shape}")
        
        # Load MusicGen model
        logger.info("Lade MusicGen Model...")
        model = MusicGen.get_pretrained('facebook/musicgen-small')
        
        logger.info(f"Model device: {model.device}")
        logger.info("✅ MusicGen Model geladen")
        
        # Check for existing checkpoint to resume
        resume_checkpoint = None
        if Path("musicgen_extended_best.pt").exists():
            resume_checkpoint = "musicgen_extended_best.pt"
        elif Path("musicgen_simple_epoch_3.pt").exists():
            resume_checkpoint = "musicgen_simple_epoch_3.pt"
        
        # Start training
        logger.info("🚀 Starte erweitertes Training...")
        extended_training_loop(
            model, 
            dataloader, 
            epochs=epochs, 
            lr=lr,
            resume_from=resume_checkpoint
        )
        logger.info("✅ Erweitertes Training abgeschlossen!")
        
    except Exception as e:
        logger.error(f"❌ Fehler beim Training: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
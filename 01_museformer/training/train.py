"""
Training Script für Museformer
Vollständiges Training mit LoRA, Checkpointing und Monitoring.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import wandb
from pathlib import Path
from tqdm.auto import tqdm
import json
import math
from typing import Dict, Optional

# Relative imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from model.config import MuseformerConfig, get_config
from model.museformer import MuseformerForCausalLM
from data.tokenizer import MIDITokenizer, create_pop_midi_tokenizer
from data.dataset import GenreConditionedMIDIDataset, create_dataloader, analyze_dataset


class MuseformerTrainer:
    """Trainer-Klasse für Museformer."""
    
    def __init__(self, 
                 config: MuseformerConfig,
                 model: MuseformerForCausalLM,
                 tokenizer: MIDITokenizer,
                 train_dataloader: DataLoader,
                 val_dataloader: Optional[DataLoader] = None,
                 output_dir: str = "./outputs",
                 use_wandb: bool = False,
                 wandb_project: str = "museformer"):
        
        self.config = config
        self.model = model
        self.tokenizer = tokenizer
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            epochs=100,  # Wird später überschrieben
            steps_per_epoch=len(train_dataloader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        # Monitoring
        self.use_wandb = use_wandb
        if use_wandb:
            wandb.init(project=wandb_project, config=config.to_dict())
            wandb.watch(self.model)
        
        self.writer = SummaryWriter(self.output_dir / "tensorboard")
        
        # Training state
        self.global_step = 0
        self.epoch = 0
        self.best_val_loss = float('inf')
        
        print(f"🚀 Museformer Trainer initialisiert")
        print(f"   Device: {self.device}")
        print(f"   Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Trainable parameters: {sum(p.numel() for p in self.model.parameters() if p.requires_grad):,}")
        print(f"   Training samples: {len(train_dataloader.dataset)}")
        if val_dataloader:
            print(f"   Validation samples: {len(val_dataloader.dataset)}")
    
    def train_epoch(self) -> Dict[str, float]:
        """Trainiert eine Epoche."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_dataloader)
        
        progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {self.epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = self.model(**batch)
            loss = outputs.loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            
            # Update metrics
            total_loss += loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{total_loss / (batch_idx + 1):.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to tensorboard/wandb
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/learning_rate', 
                                     self.scheduler.get_last_lr()[0], self.global_step)
                
                if self.use_wandb:
                    wandb.log({
                        'train/loss': loss.item(),
                        'train/learning_rate': self.scheduler.get_last_lr()[0],
                        'global_step': self.global_step
                    })
            
            # Save checkpoint periodically
            if self.global_step % 1000 == 0:
                self.save_checkpoint(f"checkpoint-step-{self.global_step}")
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    def validate(self) -> Dict[str, float]:
        """Validiert das Modell."""
        if self.val_dataloader is None:
            return {}
        
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_dataloader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Validation"):
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        perplexity = math.exp(avg_loss)
        
        # Log validation metrics
        self.writer.add_scalar('val/loss', avg_loss, self.global_step)
        self.writer.add_scalar('val/perplexity', perplexity, self.global_step)
        
        if self.use_wandb:
            wandb.log({
                'val/loss': avg_loss,
                'val/perplexity': perplexity,
                'global_step': self.global_step
            })
        
        return {'val_loss': avg_loss, 'val_perplexity': perplexity}
    
    def train(self, num_epochs: int = 10, save_every: int = 5):
        """Haupttraining-Loop."""
        print(f"🎯 Starte Training für {num_epochs} Epochen")
        
        # Update scheduler
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.learning_rate,
            epochs=num_epochs,
            steps_per_epoch=len(self.train_dataloader),
            pct_start=0.1,
            anneal_strategy='cos'
        )
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Training
            train_metrics = self.train_epoch()
            
            # Validation
            val_metrics = self.validate()
            
            # Print epoch summary
            print(f"\n📊 Epoch {epoch + 1}/{num_epochs} Summary:")
            print(f"   Train Loss: {train_metrics['train_loss']:.4f}")
            if val_metrics:
                print(f"   Val Loss: {val_metrics['val_loss']:.4f}")
                print(f"   Val Perplexity: {val_metrics['val_perplexity']:.2f}")
            
            # Save best model
            if val_metrics and val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.save_model("best_model")
                print(f"   💾 Neues bestes Modell gespeichert (Val Loss: {self.best_val_loss:.4f})")
            
            # Save periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f"checkpoint-epoch-{epoch + 1}")
        
        # Save final model
        self.save_model("final_model")
        print("✅ Training abgeschlossen!")
        
        # Close logging
        self.writer.close()
        if self.use_wandb:
            wandb.finish()
    
    def save_model(self, name: str):
        """Speichert das trainierte Modell."""
        save_path = self.output_dir / name
        save_path.mkdir(exist_ok=True)
        
        # Save model in HuggingFace format
        self.model.save_pretrained(str(save_path))
        
        # Save tokenizer
        self.tokenizer.save_vocabulary(save_path / "tokenizer.pkl")
        
        # Save training config
        with open(save_path / "training_config.json", "w") as f:
            json.dump({
                'config': self.config.to_dict(),
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_val_loss': self.best_val_loss
            }, f, indent=2)
    
    def save_checkpoint(self, name: str):
        """Speichert Training-Checkpoint."""
        checkpoint_path = self.output_dir / "checkpoints" / f"{name}.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'epoch': self.epoch,
            'best_val_loss': self.best_val_loss,
            'config': self.config.to_dict()
        }, checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path: str):
        """Lädt Training-Checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"📂 Checkpoint geladen: Step {self.global_step}, Epoch {self.epoch}")


def create_model_and_tokenizer(config_name: str = "pop_style") -> tuple:
    """Erstellt Modell und Tokenizer."""
    # Create tokenizer first
    tokenizer = create_pop_midi_tokenizer()
    
    # Load config and update with tokenizer info BEFORE creating model
    config = get_config(config_name)
    
    # CRITICAL: Update vocab_size BEFORE creating model
    original_vocab_size = config.vocab_size
    config.vocab_size = tokenizer.vocab_size
    config.pad_token_id = tokenizer.pad_token_id
    config.bos_token_id = tokenizer.bos_token_id
    config.eos_token_id = tokenizer.eos_token_id
    
    # Debug output
    print(f"🔍 Original config vocab_size: {original_vocab_size}")
    print(f"🔍 Tokenizer vocab_size: {tokenizer.vocab_size}")
    print(f"🔍 Updated config vocab_size: {config.vocab_size}")
    print(f"🔍 Special tokens: PAD={config.pad_token_id}, BOS={config.bos_token_id}, EOS={config.eos_token_id}")
    
    # Create model with correct vocab_size
    model = MuseformerForCausalLM(config)
    
    # Verify model was created successfully
    print(f"✅ Model created successfully with vocab_size={config.vocab_size}")
    
    return model, tokenizer, config


def create_datasets(tokenizer: MIDITokenizer, 
                   data_dir: str,
                   config: MuseformerConfig) -> tuple:
    """Erstellt Train- und Validation-Datasets."""
    data_dir = Path(data_dir)
    
    # Look for metadata file
    metadata_file = None
    for ext in ['.jsonl', '.json']:
        potential_file = data_dir / f"metadata{ext}"
        if potential_file.exists():
            metadata_file = potential_file
            break
    
    if metadata_file is None:
        # Fallback: Create simple dataset from MIDI files
        midi_files = list(data_dir.glob("**/*.mid")) + list(data_dir.glob("**/*.midi"))
        if not midi_files:
            raise ValueError(f"Keine MIDI-Files in {data_dir} gefunden")
        
        # Create simple metadata
        metadata_file = data_dir / "metadata.jsonl"
        with open(metadata_file, "w") as f:
            for midi_file in midi_files:
                entry = {
                    "midi_path": str(midi_file),
                    "genre": "pop"  # Default genre
                }
                f.write(json.dumps(entry) + "\n")
    
    # Create dataset
    dataset = GenreConditionedMIDIDataset(
        tokenizer=tokenizer,
        data_file=metadata_file,
        max_length=config.max_seq_length,
        cache_dir=str(data_dir / "cache")
    )
    
    # Split into train/val
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    return train_dataset, val_dataset


def main():
    parser = argparse.ArgumentParser(description="Museformer Training")
    parser.add_argument("--config", type=str, default="pop_style", 
                       help="Model configuration name")
    parser.add_argument("--data_dir", type=str, required=True,
                       help="Directory containing MIDI files")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                       help="Output directory for models and logs")
    parser.add_argument("--epochs", type=int, default=20,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use Weights & Biases for logging")
    parser.add_argument("--resume_from", type=str, default=None,
                       help="Resume training from checkpoint")
    
    args = parser.parse_args()
    
    # Create model and tokenizer
    print("🔧 Erstelle Modell und Tokenizer...")
    model, tokenizer, config = create_model_and_tokenizer(args.config)
    
    # Override config with command line args
    if args.learning_rate != 1e-4:
        config.learning_rate = args.learning_rate
    
    # Create datasets
    print("📚 Lade Datasets...")
    train_dataset, val_dataset = create_datasets(tokenizer, args.data_dir, config)
    
    # Analyze dataset
    print("📊 Dataset-Analyse:")
    if hasattr(train_dataset, 'dataset'):  # Subset wrapper
        analysis = analyze_dataset(train_dataset.dataset)
    else:
        analysis = analyze_dataset(train_dataset)
    
    for key, value in analysis.items():
        print(f"   {key}: {value}")
    
    # Create dataloaders
    train_dataloader = create_dataloader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True
    )
    
    val_dataloader = create_dataloader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    ) if val_dataset else None
    
    # Create trainer
    trainer = MuseformerTrainer(
        config=config,
        model=model,
        tokenizer=tokenizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        output_dir=args.output_dir,
        use_wandb=args.use_wandb
    )
    
    # Resume from checkpoint if specified
    if args.resume_from:
        trainer.load_checkpoint(args.resume_from)
    
    # Start training
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main() 
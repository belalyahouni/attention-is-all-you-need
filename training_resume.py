"""
Training Resume Script
====================

This script allows resuming training from a saved checkpoint.
It extends the distributed training functionality with checkpoint
loading and resumption capabilities.

Key features:
- Resume training from any saved checkpoint
- Support for both old and new checkpoint formats
- Complete state restoration (model, optimizer, scaler, step)
- Backward compatibility with older checkpoint formats
- All features from distributed_training.py

Usage:
    python training_resume.py

Requirements:
    - Multiple CUDA GPUs
    - Pre-trained tokenizer (bpe_tokenizer.json)
    - Dataset files in datasets/ directory
    - Existing checkpoint files in checkpoints/ directory

Author: Implementation for Transformer training pipeline
"""

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, distributed
import math
import os
from data_preprocessing import train_en, train_de, val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len, tokenize_sequences, load_and_prepare_data
from attention_masks import create_encoder_mask, create_decoder_mask
from transformer_model import build_transformer
# Updated autocast API
import torch.amp 

print("Starting training resume script")

import torch
from torch.utils.data import Dataset
# Import attention mask functions
from attention_masks import create_encoder_mask, create_decoder_mask

class BilingualDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, tokenizer, src_max_len, tgt_max_len):
        assert len(src_lines) == len(tgt_lines), "Source and target must have the same number of lines"
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        
        self.sos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]

        enc_input_tokens = self.tokenizer(src_text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.src_max_len - 2)['input_ids'].squeeze(0)
        dec_input_tokens = self.tokenizer(tgt_text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.tgt_max_len - 2)['input_ids'].squeeze(0)

        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            enc_input_tokens.long(), # <--- FIX: Ensure long type
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.tensor([self.pad_token_id] * (self.src_max_len - len(enc_input_tokens) - 2), dtype=torch.long)
        ])

        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            dec_input_tokens.long(), # <--- FIX: Ensure long type
            torch.tensor([self.pad_token_id] * (self.tgt_max_len - len(dec_input_tokens) - 1), dtype=torch.long)
        ])
        
        label = torch.cat([
            dec_input_tokens.long(), # <--- FIX: Ensure long type
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.tensor([self.pad_token_id] * (self.tgt_max_len - len(dec_input_tokens) - 1), dtype=torch.long)
        ])

        assert encoder_input.size(0) == self.src_max_len
        assert decoder_input.size(0) == self.tgt_max_len
        assert label.size(0) == self.tgt_max_len

        encoder_mask = create_encoder_mask(encoder_input, self.pad_token_id)
        decoder_mask = create_decoder_mask(decoder_input, self.pad_token_id)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask
        }

# ---------------- DDP setup ----------------
def setup_ddp(rank, world_size):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

# ---------------- Training function ----------------
def train_ddp(rank, world_size, train_en, train_de, val_en, val_de):
    print(f"Running DDP training on rank {rank}.")
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # --- GRADIENT ACCUMULATION SETUP ---
    effective_batch_size = 164
    physical_batch_size = 64
    accumulation_steps = effective_batch_size // physical_batch_size
    
    # --- CHECKPOINT CONFIGURATION ---
    # Set to None to start from scratch, or a path to resume training.
    # For example: "checkpoints/step_10000.pt"
    resume_from_checkpoint = "checkpoints/step_240000.pt" # <--- CHANGE THIS LINE
    # --------------------------------

    train_dataset = BilingualDataset(train_en, train_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
    val_dataset = BilingualDataset(val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    train_loader = DataLoader(
        train_dataset, batch_size=physical_batch_size, sampler=train_sampler,
        num_workers=8, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=physical_batch_size, num_workers=8, pin_memory=True
    )

    # Build model and move to device BEFORE loading state_dict
    vocab_size = bpe_tokenizer.vocab_size
    transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len).to(device)
    
    # Optimizer and loss function
    d_model = 512
    warmup_steps = 5000
    optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=bpe_tokenizer.pad_token_id).to(device)
    scaler = torch.amp.GradScaler("cuda")
    
    step = 0 # Default starting step
    
    # --- NEW: LOAD CHECKPOINT IF SPECIFIED ---
    if resume_from_checkpoint and os.path.exists(resume_from_checkpoint):
        print(f"[Rank {rank}] Loading checkpoint: {resume_from_checkpoint}")
        loc = f'cuda:{rank}'
        checkpoint = torch.load(resume_from_checkpoint, map_location=loc)
        
        # --- START: BACKWARD COMPATIBILITY FIX ---
        # Check if the checkpoint is a dictionary with the expected key.
        # This handles both new and old checkpoint formats.
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            # New format: Load model, optimizer, scaler, and step
            print(f"[Rank {rank}] Loading from new checkpoint format.")
            transformer.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
            step = checkpoint['step']
        else:
            # Old format: Checkpoint is just the model state_dict
            print(f"[Rank {rank}] WARNING: Loading from old checkpoint format.")
            print(f"[Rank {rank}] Only model weights will be restored. Optimizer and scaler state are reset.")
            transformer.load_state_dict(checkpoint)
            # Try to infer step from filename, otherwise start from 0 for LR scheduling
            try:
                step = int(os.path.basename(resume_from_checkpoint).split('_')[-1].split('.')[0])
                print(f"[Rank {rank}] Inferred step {step} from filename for learning rate.")
            except:
                step = 0 # Fallback if filename parsing fails
                print(f"[Rank {rank}] Could not infer step from filename. Step count reset to 0.")
        # --- END: BACKWARD COMPATIBILITY FIX ---
        
        print(f"[Rank {rank}] Resuming training from step {step}")
    else:
        print(f"[Rank {rank}] Starting training from scratch.")
    
    # IMPORTANT: Wrap model with DDP *after* loading the state_dict
    transformer = nn.parallel.DistributedDataParallel(transformer, device_ids=[rank])
    
    def lr_scheduler(step):
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    num_steps = 1400000
    save_interval = 20000
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # The training loop now starts from the resumed 'step'
    while step < num_steps:
        train_sampler.set_epoch(step)
        
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            transformer.train()
            
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                enc_out = transformer.module.encode(encoder_input, encoder_mask)
                dec_out = transformer.module.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
                logits = transformer.module.project(dec_out)
                loss = criterion(logits.view(-1, vocab_size), label.view(-1))
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0:
                for param_group in optimizer.param_groups:
                    # Update LR based on the *new* step number
                    param_group['lr'] = lr_scheduler(step + 1)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                
                step += 1 # Increment step after a full optimizer update

                if rank == 0 and step % save_interval == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                    # --- NEW: SAVE COMPLETE STATE DICTIONARY ---
                    torch.save({
                        'step': step,
                        'model_state_dict': transformer.module.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scaler_state_dict': scaler.state_dict(),
                    }, ckpt_path)
                    # -------------------------------------------
                    print(f"[Rank {rank}] Saved checkpoint at step {step}, loss: {loss.item() * accumulation_steps:.4f}")

            if step >= num_steps:
                break
        
        if step >= num_steps:
            break

    cleanup_ddp()
    print(f"Rank {rank} training complete!")

# ---------------- Main entry ----------------
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    
    train_en, train_de, val_en, val_de = load_and_prepare_data()
    
    args = (world_size, train_en, train_de, val_en, val_de)
    mp.spawn(train_ddp, args=args, nprocs=world_size, join=True)
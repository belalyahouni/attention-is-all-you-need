"""
Distributed Training Script
==========================

This script implements multi-GPU distributed training for the Transformer model.
It uses PyTorch's DistributedDataParallel (DDP) to train the model across multiple GPUs.

Key features:
- Multi-GPU training with DDP
- Gradient accumulation for larger effective batch sizes
- Mixed precision training (bfloat16)
- Learning rate scheduling with warmup
- Checkpoint saving and resumption
- Label smoothing for better generalization

Usage:
    python distributed_training.py

Requirements:
    - Multiple CUDA GPUs
    - Pre-trained tokenizer (bpe_tokenizer.json)
    - Dataset files in datasets/ directory

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
from torch.cuda.amp import GradScaler, autocast

print("Starting distributed training script")

import torch
from torch.utils.data import Dataset
# Import attention mask functions
from attention_masks import create_encoder_mask, create_decoder_mask

class BilingualDataset(Dataset):
    """
    PyTorch Dataset for parallel bilingual text data.
    
    This dataset handles the conversion of raw text pairs into tensors
    that can be fed to the Transformer model. It handles tokenization,
    padding, and mask creation.
    
    Args:
        src_lines: List of source language sentences
        tgt_lines: List of target language sentences
        tokenizer: PreTrainedTokenizerFast instance
        src_max_len: Maximum source sequence length
        tgt_max_len: Maximum target sequence length
    """
    
    def __init__(self, src_lines, tgt_lines, tokenizer, src_max_len, tgt_max_len):
        # Ensure source and target have the same number of sentences
        assert len(src_lines) == len(tgt_lines), "Source and target must have the same number of lines"
        
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        
        # Cache special token IDs for efficiency
        self.sos_token_id = tokenizer.bos_token_id  # Start of sequence
        self.eos_token_id = tokenizer.eos_token_id  # End of sequence
        self.pad_token_id = tokenizer.pad_token_id  # Padding token

    def __len__(self):
        """Return the number of sentence pairs in the dataset."""
        return len(self.src_lines)

    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Args:
            idx: Index of the example to retrieve
            
        Returns:
            Dictionary containing:
            - encoder_input: Source tokens with special tokens
            - decoder_input: Target tokens with <bos> token
            - label: Target tokens with <eos> token (for loss calculation)
            - encoder_mask: Mask for encoder attention
            - decoder_mask: Mask for decoder attention
        """
        # Get source and target sentences
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]

        # Tokenize source and target texts (without special tokens initially)
        enc_input_tokens = self.tokenizer(
            src_text, 
            add_special_tokens=False, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.src_max_len - 2  # Reserve space for <bos> and <eos>
        )['input_ids'].squeeze(0)
        
        dec_input_tokens = self.tokenizer(
            tgt_text, 
            add_special_tokens=False, 
            return_tensors="pt", 
            truncation=True, 
            max_length=self.tgt_max_len - 2  # Reserve space for <bos> and <eos>
        )['input_ids'].squeeze(0)

        # Prepare encoder input: <bos> + source_text + <eos> + <pad>...
        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            enc_input_tokens,
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.tensor([self.pad_token_id] * (self.src_max_len - len(enc_input_tokens) - 2), dtype=torch.long)
        ])

        # Prepare decoder input: <bos> + target_text + <pad>...
        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            dec_input_tokens,
            torch.tensor([self.pad_token_id] * (self.tgt_max_len - len(dec_input_tokens) - 1), dtype=torch.long)
        ])
        
        # Prepare the label (target for loss calculation): target_text + <eos> + <pad>...
        label = torch.cat([
            dec_input_tokens,
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.tensor([self.pad_token_id] * (self.tgt_max_len - len(dec_input_tokens) - 1), dtype=torch.long)
        ])

        # Verify tensor dimensions
        assert encoder_input.size(0) == self.src_max_len
        assert decoder_input.size(0) == self.tgt_max_len
        assert label.size(0) == self.tgt_max_len

        # Create attention masks
        encoder_mask = create_encoder_mask(encoder_input, self.pad_token_id)
        decoder_mask = create_decoder_mask(decoder_input, self.pad_token_id)

        return {
            "encoder_input": encoder_input,
            "decoder_input": decoder_input,
            "label": label,
            "encoder_mask": encoder_mask,
            "decoder_mask": decoder_mask
        }

# ============================================================================
# DISTRIBUTED DATA PARALLEL SETUP
# ============================================================================

def setup_ddp(rank, world_size):
    """
    Initialize distributed data parallel training.
    
    This function sets up the distributed training environment,
    including process group initialization and GPU assignment.
    
    Args:
        rank (int): Process rank (0 to world_size-1)
        world_size (int): Total number of processes
    """
    # Set environment variables for distributed training
    os.environ['MASTER_ADDR'] = '127.0.0.1'  # Master node address
    os.environ['MASTER_PORT'] = '29500'      # Master node port
    
    # Initialize the process group
    dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    
    # Set the current GPU device
    torch.cuda.set_device(rank)

def cleanup_ddp():
    """
    Clean up distributed training resources.
    
    This function destroys the process group and frees up resources
    when training is complete.
    """
    dist.destroy_process_group()

# ============================================================================
# TRAINING FUNCTION
# ============================================================================

# Import updated autocast API
torch.amp

def train_ddp(rank, world_size, train_en, train_de, val_en, val_de):
    """
    Main training function for distributed data parallel training.
    
    This function runs on each GPU process and handles:
    - Model initialization and DDP wrapping
    - Data loading with distributed sampling
    - Training loop with gradient accumulation
    - Mixed precision training
    - Learning rate scheduling
    - Checkpoint saving
    
    Args:
        rank (int): Process rank (0 to world_size-1)
        world_size (int): Total number of processes
        train_en: English training sentences
        train_de: German training sentences
        val_en: English validation sentences
        val_de: German validation sentences
    """
    print(f"Running DDP training on rank {rank}.")
    setup_ddp(rank, world_size)

    # Set device for this process
    device = torch.device(f"cuda:{rank}")

    # ============================================================================
    # GRADIENT ACCUMULATION CONFIGURATION
    # ============================================================================
    # Gradient accumulation allows us to simulate larger batch sizes
    # by accumulating gradients over multiple mini-batches before updating
    effective_batch_size = 64      # Desired effective batch size
    physical_batch_size = 32       # Actual batch size that fits in GPU memory
    accumulation_steps = effective_batch_size // physical_batch_size  # Steps to accumulate
    
    print(f"Gradient accumulation: {accumulation_steps} steps of {physical_batch_size} = {effective_batch_size} effective batch size")

    # ============================================================================
    # DATA LOADING AND DATASET PREPARATION
    # ============================================================================
    
    # Create datasets
    train_dataset = BilingualDataset(train_en, train_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
    val_dataset = BilingualDataset(val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
    
    # Create distributed sampler for training data
    # This ensures each process sees different data
    train_sampler = distributed.DistributedSampler(
        train_dataset, 
        num_replicas=world_size, 
        rank=rank, 
        shuffle=True
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        sampler=train_sampler,
        num_workers=8,        # Number of worker processes for data loading
        pin_memory=True        # Faster GPU transfer
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=physical_batch_size,
        num_workers=8,
        pin_memory=True
    )

    # ============================================================================
    # MODEL INITIALIZATION
    # ============================================================================
    
    # Build the Transformer model
    vocab_size = bpe_tokenizer.vocab_size
    transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len).to(device)
    
    # Wrap model with DistributedDataParallel
    transformer = nn.parallel.DistributedDataParallel(transformer, device_ids=[rank])

    # ============================================================================
    # OPTIMIZER AND LOSS FUNCTION SETUP
    # ============================================================================
    
    # Model hyperparameters
    d_model = 512
    warmup_steps = 5000
    
    # Adam optimizer with parameters from the original paper
    optimizer = torch.optim.Adam(
        transformer.parameters(), 
        betas=(0.9, 0.98),  # Beta parameters for momentum
        eps=1e-9           # Epsilon for numerical stability
    )
    
    # Cross-entropy loss with label smoothing and padding token ignored
    criterion = nn.CrossEntropyLoss(
        label_smoothing=0.1,                    # Label smoothing for better generalization
        ignore_index=bpe_tokenizer.pad_token_id # Ignore padding tokens in loss
    ).to(device)
    
    # Mixed precision scaler for gradient scaling
    scaler = torch.amp.GradScaler("cuda")

    # ============================================================================
    # LEARNING RATE SCHEDULING
    # ============================================================================
    
    def lr_scheduler(step):
        """
        Learning rate scheduler following the Transformer paper.
        
        Implements the learning rate schedule from "Attention Is All You Need":
        lr = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
        
        Args:
            step: Current training step
            
        Returns:
            Learning rate for this step
        """
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    # ============================================================================
    # TRAINING CONFIGURATION
    # ============================================================================
    
    num_steps = 100000      # Total number of training steps
    save_interval = 1000   # Save checkpoint every N steps
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0  # Current training step
    # ============================================================================
    # MAIN TRAINING LOOP
    # ============================================================================
    
    while step < num_steps:
        # Set epoch for distributed sampler (ensures different data each epoch)
        train_sampler.set_epoch(step)
        
        # Zero gradients at the start of each accumulation cycle
        optimizer.zero_grad(set_to_none=True)

        # Process batches in the current epoch
        for i, batch in enumerate(train_loader):
            transformer.train()  # Set model to training mode
            
            # Move batch data to GPU
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # Forward pass with mixed precision
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                # Encode source sequence
                enc_out = transformer.module.encode(encoder_input, encoder_mask)
                # Decode target sequence
                dec_out = transformer.module.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
                # Project to vocabulary logits
                logits = transformer.module.project(dec_out)
                # Calculate loss
                loss = criterion(logits.view(-1, vocab_size), label.view(-1))
                
                # Normalize loss for gradient accumulation
                loss = loss / accumulation_steps

            # Backward pass with gradient scaling
            scaler.scale(loss).backward()

            # ============================================================================
            # GRADIENT ACCUMULATION AND OPTIMIZER UPDATE
            # ============================================================================
            
            # Check if we have accumulated enough gradients
            if (i + 1) % accumulation_steps == 0:
                # Update learning rate before optimizer step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler(step + 1)

                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                
                # Reset gradients for next accumulation cycle
                optimizer.zero_grad(set_to_none=True)
                
                # Increment training step
                step += 1

                # Save checkpoint periodically (only on rank 0)
                if rank == 0 and step % save_interval == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                    torch.save(transformer.module.state_dict(), ckpt_path)
                    # Note: Loss is normalized, so multiply by accumulation_steps for true loss
                    print(f"[Rank {rank}] Saved checkpoint at step {step}, loss: {loss.item() * accumulation_steps:.4f}")

            # Check if we've reached the maximum number of steps
            if step >= num_steps:
                break
        
        # Break outer loop if we've reached max steps
        if step >= num_steps:
            break

    # ============================================================================
    # CLEANUP
    # ============================================================================
    
    cleanup_ddp()
    print(f"Rank {rank} training complete!")

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Main entry point for distributed training.
    
    This script automatically detects the number of available GPUs
    and spawns one process per GPU for distributed training.
    """
    # Get number of available GPUs
    world_size = torch.cuda.device_count()
    print(f"Starting distributed training on {world_size} GPUs")
    
    # Load data once in the main process
    # This avoids loading data multiple times across processes
    train_en, train_de, val_en, val_de = load_and_prepare_data()
    
    # Prepare arguments for spawned processes
    args = (world_size, train_en, train_de, val_en, val_de)
    
    # Spawn processes for distributed training
    mp.spawn(train_ddp, args=args, nprocs=world_size, join=True)
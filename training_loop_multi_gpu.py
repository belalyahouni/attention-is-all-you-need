import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset, distributed
import math
import os
from tokenize_datasets import train_en, train_de, val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len, tokenize_sequences, load_and_prepare_data
from create_masks import create_encoder_mask, create_decoder_mask
from model import build_transformer
from torch.cuda.amp import GradScaler, autocast

print("Starting file")

import torch
from torch.utils.data import Dataset
# Make sure you import create_encoder_mask and create_decoder_mask
from create_masks import create_encoder_mask, create_decoder_mask

class BilingualDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, tokenizer, src_max_len, tgt_max_len):
        assert len(src_lines) == len(tgt_lines), "Source and target must have the same number of lines"
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len
        
        # Cache special token IDs
        self.sos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        src_text = self.src_lines[idx]
        tgt_text = self.tgt_lines[idx]

        # Tokenize source and target texts
        # We don't add special tokens yet, we'll do that manually
        enc_input_tokens = self.tokenizer(src_text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.src_max_len - 2)['input_ids'].squeeze(0)
        dec_input_tokens = self.tokenizer(tgt_text, add_special_tokens=False, return_tensors="pt", truncation=True, max_length=self.tgt_max_len - 2)['input_ids'].squeeze(0)

        # Prepare encoder input: <s> + source_text + </s> + <pad>...
        encoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            enc_input_tokens,
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.tensor([self.pad_token_id] * (self.src_max_len - len(enc_input_tokens) - 2), dtype=torch.long)
        ])

        # Prepare decoder input: <s> + target_text + <pad>...
        decoder_input = torch.cat([
            torch.tensor([self.sos_token_id], dtype=torch.long),
            dec_input_tokens,
            torch.tensor([self.pad_token_id] * (self.tgt_max_len - len(dec_input_tokens) - 1), dtype=torch.long)
        ])
        
        # Prepare the label (the target): target_text + </s> + <pad>...
        label = torch.cat([
            dec_input_tokens,
            torch.tensor([self.eos_token_id], dtype=torch.long),
            torch.tensor([self.pad_token_id] * (self.tgt_max_len - len(dec_input_tokens) - 1), dtype=torch.long)
        ])

        # Ensure all tensors have the expected max length
        assert encoder_input.size(0) == self.src_max_len
        assert decoder_input.size(0) == self.tgt_max_len
        assert label.size(0) == self.tgt_max_len

        # Create masks for the encoder and decoder.
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
# Add this new import at the top of your training script
import torch.amp

# ---------------- Training function ----------------
def train_ddp(rank, world_size, train_en, train_de, val_en, val_de):
    print(f"Running DDP training on rank {rank}.")
    setup_ddp(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    # --- GRADIENT ACCUMULATION SETUP ---
    # 1. Set the desired effective batch size
    effective_batch_size = 64
    # 2. Choose a smaller physical batch size that fits in memory
    physical_batch_size = 32  # Try 32 or 64
    # 3. Calculate accumulation steps
    accumulation_steps = effective_batch_size // physical_batch_size
    # -----------------------------------

    train_dataset = BilingualDataset(train_en, train_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
    val_dataset = BilingualDataset(val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
    train_sampler = distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    
    # Use the smaller physical_batch_size here
    train_loader = DataLoader(
        train_dataset,
        batch_size=physical_batch_size,
        sampler=train_sampler,
        num_workers=8,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=physical_batch_size,
        num_workers=8,
        pin_memory=True
    )

    # Build model
    vocab_size = bpe_tokenizer.vocab_size
    transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len).to(device)
    transformer = nn.parallel.DistributedDataParallel(transformer, device_ids=[rank])

    d_model = 512
    warmup_steps = 5000
    optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1, ignore_index=bpe_tokenizer.pad_token_id).to(device)
    
    # API FIX: Use the updated API to remove the FutureWarning
    scaler = torch.amp.GradScaler("cuda")

    def lr_scheduler(step):
        return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

    num_steps = 100000
    save_interval = 1000
    checkpoint_dir = "checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)

    step = 0
    while step < num_steps:
        train_sampler.set_epoch(step)
        
        # Zero gradients once before the accumulation loop
        optimizer.zero_grad(set_to_none=True)

        for i, batch in enumerate(train_loader):
            transformer.train()
            
            encoder_input = batch['encoder_input'].to(device)
            decoder_input = batch['decoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            decoder_mask = batch['decoder_mask'].to(device)
            label = batch['label'].to(device)
            
            # API FIX: Use updated autocast API
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
                enc_out = transformer.module.encode(encoder_input, encoder_mask)
                dec_out = transformer.module.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
                logits = transformer.module.project(dec_out)
                loss = criterion(logits.view(-1, vocab_size), label.view(-1))
                
                # Normalize loss to account for accumulation
                loss = loss / accumulation_steps

            scaler.scale(loss).backward()

            # --- MODEL UPDATE STEP ---
            # Check if we have accumulated enough gradients
            if (i + 1) % accumulation_steps == 0:
                # Update learning rate just before the optimizer step
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr_scheduler(step + 1)

                scaler.step(optimizer)
                scaler.update()
                
                # Reset gradients for the next accumulation cycle
                optimizer.zero_grad(set_to_none=True)
                
                step += 1

                if rank == 0 and step % save_interval == 0:
                    ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
                    torch.save(transformer.module.state_dict(), ckpt_path)
                    # Note: The printed loss is the normalized loss of the last mini-batch
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
    
    # NEW: Load data ONCE here in the main process
    train_en, train_de, val_en, val_de = load_and_prepare_data()
    
    # NEW: Pass the loaded data as arguments to the spawned processes
    args = (world_size, train_en, train_de, val_en, val_de)
    mp.spawn(train_ddp, args=args, nprocs=world_size, join=True)
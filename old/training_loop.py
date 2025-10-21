import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import math
import os
from tokenize_datasets import train_en, train_de, val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len, tokenize_sequences
from create_masks import create_encoder_mask, create_decoder_mask
from model import build_transformer

print("Starting file")
class BilingualDataset(Dataset):
    def __init__(self, src_lines, tgt_lines, tokenizer, src_max_len, tgt_max_len):
        assert len(src_lines) == len(tgt_lines), "Source and target must have same number of lines"
        self.src_lines = src_lines
        self.tgt_lines = tgt_lines
        self.tokenizer = tokenizer
        self.src_max_len = src_max_len
        self.tgt_max_len = tgt_max_len

        # Pre-tokenize and store as tensors
        self.encoder_inputs = tokenize_sequences(src_lines, tokenizer, src_max_len)['input_ids']
        self.decoder_inputs = tokenize_sequences(tgt_lines, tokenizer, tgt_max_len)['input_ids']
        self.labels = self.decoder_inputs.clone()  # labels = decoder inputs

        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.src_lines)

    def __getitem__(self, idx):
        encoder_input = self.encoder_inputs[idx]
        decoder_input = self.decoder_inputs[idx]
        label = self.labels[idx]

        # Masks
        encoder_mask = create_encoder_mask(encoder_input, self.pad_token_id)
        decoder_mask = create_decoder_mask(decoder_input, self.pad_token_id)

        return {
            "encoder_input": encoder_input,    # (seq_len)
            "decoder_input": decoder_input,    # (seq_len)
            "label": label,                    # (seq_len)
            "encoder_mask": encoder_mask,      # (1,1,seq_len)
            "decoder_mask": decoder_mask       # (1, seq_len, seq_len)
        }


train_dataset = BilingualDataset(train_en, train_de, bpe_tokenizer, src_seq_len, tgt_seq_len)
val_dataset = BilingualDataset(val_en, val_de, bpe_tokenizer, src_seq_len, tgt_seq_len)

print("Loaded BilingualDatasets")
train_loader = DataLoader(train_dataset, batch_size=100, shuffle=True)  # batch=1 because each sample ~25k tokens
val_loader = DataLoader(val_dataset, batch_size=100)
print("Loaded DataLoaded")
vocab_size = bpe_tokenizer.vocab_size

transformer = build_transformer(vocab_size, vocab_size, src_seq_len, tgt_seq_len)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Wrap with DataParallel
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
    transformer = nn.DataParallel(transformer)

transformer.to(device)
transformer.to(device)
print("Built Transformer")
d_model = 512
warmup_steps = 500

optimizer = torch.optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-9)

def lr_scheduler(step):
    return (d_model ** -0.5) * min(step ** -0.5, step * (warmup_steps ** -1.5))

class LabelSmoothingLoss(nn.Module):
    def __init__(self, label_smoothing, tgt_vocab_size, ignore_index=-100):
        super().__init__()
        self.smoothing = label_smoothing
        self.vocab_size = tgt_vocab_size
        self.ignore_index = ignore_index

    def forward(self, pred, target):
        # pred: [batch, seq_len, vocab], target: [batch, seq_len]
        pred = pred.view(-1, self.vocab_size)
        target = target.view(-1)

        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.vocab_size - 1))
            ignore = target == self.ignore_index
            target_clamped = target.clone()
            target_clamped[ignore] = 0
            true_dist.scatter_(1, target_clamped.unsqueeze(1), 1.0 - self.smoothing)
            true_dist[ignore] = 0

        loss = torch.nn.functional.kl_div(pred.log_softmax(dim=-1), true_dist, reduction='sum')
        return loss / target.size(0)

criterion = LabelSmoothingLoss(label_smoothing=0.1, tgt_vocab_size=vocab_size, ignore_index=bpe_tokenizer.pad_token_id)

num_steps = 100000
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)
save_interval = 600  # save every 600 steps (~10 min)

step = 0
while step < num_steps:
    for batch in train_loader:
        transformer.train()
        optimizer.zero_grad()

        # Move to device
        encoder_input = batch['encoder_input'].to(device)
        decoder_input = batch['decoder_input'].to(device)
        encoder_mask = batch['encoder_mask'].to(device)
        decoder_mask = batch['decoder_mask'].to(device)
        label = batch['label'].to(device)

        # Forward
        enc_out = transformer.module.encode(encoder_input, encoder_mask)
        dec_out = transformer.module.decode(enc_out, encoder_mask, decoder_input, decoder_mask)
        logits = transformer.module.projection(dec_out)

        # Compute loss
        loss = criterion(logits, label)
        loss.backward()

        # Update learning rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_scheduler(step + 1)

        optimizer.step()
        step += 1

        # Save checkpoint
        if step % save_interval == 0:
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}.pt")
            torch.save(transformer.state_dict(), ckpt_path)
            print(f"Saved checkpoint at step {step}, loss: {loss.item():.4f}")

        # Stop if reached num_steps
        if step >= num_steps:
            break

print("Training complete!")

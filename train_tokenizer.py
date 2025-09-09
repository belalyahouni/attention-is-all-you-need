from datasets import load_dataset

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

from transformers import PreTrainedTokenizerFast

# 1. Load dataset
dataset = load_dataset("wmt14", "de-en")

# 2. Extract English and German training data
train_pairs = [
    (x["translation"]["en"], x["translation"]["de"])
    for x in dataset["train"]
    if x["translation"]["en"].strip() != "" and x["translation"]["de"].strip() != ""
]

# Limit to 4.5M
train_pairs = train_pairs[:1000]

# Now split
train_en, train_de = zip(*train_pairs)
print(len(train_en), len(train_de))

validation_en = [x["translation"]["en"] for x in dataset["validation"]]
validation_de = [x["translation"]["de"] for x in dataset["validation"]]

test_en = [x["translation"]["en"] for x in dataset["test"]]
test_de = [x["translation"]["de"] for x in dataset["test"]]

# 3. Save to text files (required for tokenizer training)

with open("datasets/train.en", "w", encoding="utf-8") as f:
    for line in train_en:
        f.write(line + "\n")

with open("datasets/train.de", "w", encoding="utf-8") as f:
    for line in train_de:
        f.write(line + "\n")

with open("datasets/validation.en", "w", encoding="utf-8") as f:
    for line in validation_en:
        f.write(line + "\n")

with open("datasets/validation.de", "w", encoding="utf-8") as f:
    for line in validation_de:
        f.write(line + "\n")

with open("datasets/test.en", "w", encoding="utf-8") as f:
    for line in test_en:
        f.write(line + "\n")

with open("datasets/test.de", "w", encoding="utf-8") as f:
    for line in test_de:
        f.write(line + "\n")


# 4. Initialize tokenizer
tokenizer = Tokenizer(BPE(unk_token="<unk>"))
tokenizer.pre_tokenizer = Whitespace()

trainer = BpeTrainer(
    vocab_size=37000,
    special_tokens=["<unk>", "<pad>", "<bos>", "<eos>"]
)

# 5. Train tokenizer on both languages jointly
tokenizer.train(["datasets/train.en", "datasets/train.de"], trainer)

# 6. Save tokenizer
tokenizer.save("bpe_tokenizer.json")

# 7. Wrap tokenizer with Hugging Face for easier use
bpe_tokenizer = PreTrainedTokenizerFast(tokenizer_file="bpe_tokenizer.json")
bpe_tokenizer.add_special_tokens({"bos_token": "<bos>", "eos_token": "<eos>", "pad_token": "<pad>"})

# 8. Example encoding/decoding
encoded = bpe_tokenizer("Machine translation with transformers.")
print("IDs:", encoded.input_ids)
print("Decoded:", bpe_tokenizer.decode(encoded.input_ids))

print(len(train_en), len(train_de))

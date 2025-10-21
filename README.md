# attention-is-all-you-need
This is a reimplementation of the research paper Attention is All You Need, creating a transformer model from scratch.

1. Run train_tokenizer.py to download datasets, and train bpe tokenizer on them.
2. Run training_loop_multi_gpu.py to tokenize the dataset, and train the model in model.py
3. Run test.py (adjust input file) to test inference on model.
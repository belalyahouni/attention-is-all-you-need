# attention-is-all-you-need
This is a reimplementation of the research paper Attention is All You Need, creating a transformer model from scratch.

1. Run tokenizer_training.py to download datasets, and train bpe tokenizer on them.
2. Run distributed_training.py to tokenize the dataset, and train the model in transformer_model.py
3. Run model_evaluation.py (adjust input file) to test inference on model.
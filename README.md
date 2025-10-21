# Attention Is All You Need - Transformer Implementation

A complete, educational implementation of the Transformer architecture from the groundbreaking paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) by Vaswani et al. This project provides a hands-on learning experience for understanding how Transformer models work internally, from data preparation to model evaluation.

## 🎯 Project Purpose

This project was created to gain a deep understanding of:
- **Transformer Architecture**: How attention mechanisms work in practice
- **Machine Translation Pipeline**: Complete workflow from raw text to translated output
- **Deep Learning Implementation**: Building complex models from scratch
- **Production-Ready Code**: Professional practices for ML projects

## 📚 Educational Value

This implementation is designed for students and researchers who want to:
- Understand the inner workings of Transformer models
- Learn how to implement attention mechanisms from scratch
- Experience the complete ML pipeline (data → training → evaluation)
- See how modern NLP models are built and trained
- Gain hands-on experience with distributed training

## 🏗️ Project Architecture

### Core Components

```
📁 Project Structure
├── 🧠 transformer_model.py      # Complete Transformer architecture
├── 🎭 attention_masks.py        # Attention masking utilities
├── 📊 data_preprocessing.py     # Data loading and preparation
├── 🚀 distributed_training.py  # Multi-GPU training script
├── 🔄 training_resume.py       # Checkpoint resumption
├── 📈 model_evaluation.py      # BLEU score evaluation
├── 🔍 beam_search_inference.py # Advanced decoding
├── ⚡ model_inference.py        # Basic inference
├── 🔧 tokenizer_training.py    # BPE tokenizer training
├── 📋 batch_evaluation.sh      # Automated evaluation
└── 📁 old/                     # Legacy implementations
```

### Model Architecture

The Transformer model consists of:

1. **Input Embeddings**: Convert tokens to dense vectors
2. **Positional Encoding**: Add position information to embeddings
3. **Multi-Head Attention**: Core attention mechanism
4. **Feed-Forward Networks**: Position-wise transformations
5. **Layer Normalization**: Stabilize training
6. **Residual Connections**: Enable deep networks
7. **Encoder-Decoder Structure**: Process source and generate target

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended for training)
- 8GB+ RAM
- 10GB+ disk space for datasets

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd attention-is-all-you-need
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   ```

## 📋 Complete Workflow

### Step 1: Data Preparation and Tokenization

```bash
python tokenizer_training.py
```

**What this does:**
- Downloads WMT14 English-German dataset (4.5M sentence pairs)
- Trains a BPE tokenizer with 37,000 vocabulary size
- Saves datasets to text files for training
- Creates `bpe_tokenizer.json` for model use

**Learning outcomes:**
- Understanding tokenization strategies
- BPE algorithm implementation
- Data preprocessing pipelines

### Step 2: Model Training

```bash
python distributed_training.py
```

**What this does:**
- Implements multi-GPU distributed training
- Uses gradient accumulation for larger effective batch sizes
- Applies mixed precision training (bfloat16)
- Implements learning rate scheduling with warmup
- Saves checkpoints every 1,000 steps

**Key features:**
- **DistributedDataParallel**: Scales across multiple GPUs
- **Gradient Accumulation**: Simulates larger batch sizes
- **Mixed Precision**: Faster training with less memory
- **Learning Rate Scheduling**: Follows Transformer paper exactly
- **Label Smoothing**: Improves generalization

**Learning outcomes:**
- Distributed training concepts
- Gradient accumulation techniques
- Mixed precision training
- Learning rate scheduling strategies

### Step 3: Model Evaluation

```bash
python model_evaluation.py step_100000.pt
```

**What this does:**
- Loads trained model from checkpoint
- Translates test sentences using greedy decoding
- Calculates BLEU scores using SacreBLEU
- Provides detailed evaluation metrics

**Learning outcomes:**
- Model evaluation methodologies
- BLEU score calculation
- Translation quality assessment

### Step 4: Advanced Inference (Optional)

```bash
python beam_search_inference.py
```

**What this does:**
- Implements beam search decoding
- Explores multiple translation paths
- Applies length penalty for better quality
- Often produces higher quality translations

**Learning outcomes:**
- Beam search algorithm implementation
- Decoding strategies comparison
- Translation quality optimization

## 🔬 Technical Deep Dive

### Attention Mechanism

The core innovation of Transformers is the **Multi-Head Attention** mechanism:

```python
# Simplified attention computation
attention_scores = (Q @ K.T) / sqrt(d_k)
attention_weights = softmax(attention_scores)
output = attention_weights @ V
```

**Key concepts:**
- **Query (Q)**: What we're looking for
- **Key (K)**: What we're matching against
- **Value (V)**: The actual content we retrieve
- **Scaled Dot-Product**: Prevents gradient vanishing

### Positional Encoding

Since Transformers have no inherent notion of position, we add positional information:

```python
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

**Why this works:**
- Unique encoding for each position
- Relative position information preserved
- Allows model to understand sequence order

### Training Details

**Model Configuration:**
- **d_model**: 512 (embedding dimension)
- **N**: 6 (encoder/decoder layers)
- **h**: 8 (attention heads)
- **d_ff**: 2048 (feed-forward dimension)
- **dropout**: 0.1
- **vocab_size**: 37,000

**Training Configuration:**
- **Optimizer**: Adam (β₁=0.9, β₂=0.98, ε=1e-9)
- **Learning Rate**: Warmup for 5,000 steps, then decay
- **Batch Size**: Effective 64 (with gradient accumulation)
- **Loss**: Cross-entropy with label smoothing (0.1)

## 📊 Results and Performance

### Training Progress

The model shows typical training behavior:
- **Early steps**: Rapid improvement in BLEU scores
- **Mid training**: Gradual refinement
- **Later steps**: Convergence with occasional fluctuations

### Evaluation Results

Based on `bleu_scores.csv`:
- **Best BLEU score**: 14.21 at step 100,000
- **Performance varies**: Shows importance of checkpoint selection
- **Beam search**: Typically improves scores by 1-2 points

## 🛠️ Advanced Usage

### Resume Training

```bash
python training_resume.py
```

**Features:**
- Resume from any checkpoint
- Complete state restoration
- Backward compatibility with old checkpoints

### Batch Evaluation

```bash
bash batch_evaluation.sh
```

**Features:**
- Automated evaluation of multiple checkpoints
- CSV output for analysis
- Progress tracking and error handling

### Custom Configuration

You can modify training parameters in the scripts:

```python
# In distributed_training.py
effective_batch_size = 64      # Adjust based on GPU memory
physical_batch_size = 32       # Actual batch size per GPU
num_steps = 100000            # Total training steps
save_interval = 1000          # Checkpoint frequency
```

## 🎓 Learning Outcomes

After completing this project, you will understand:

### Technical Skills
- **Transformer Architecture**: Complete implementation from scratch
- **Attention Mechanisms**: Multi-head attention and its variants
- **Distributed Training**: Multi-GPU training strategies
- **Mixed Precision**: Memory-efficient training techniques
- **Evaluation Metrics**: BLEU scores and translation quality

### Practical Skills
- **ML Pipeline**: End-to-end machine learning workflow
- **Code Organization**: Professional project structure
- **Documentation**: Clear, educational code comments
- **Version Control**: Managing model checkpoints
- **Performance Optimization**: Training efficiency techniques

### Research Understanding
- **Paper Implementation**: Translating research to code
- **Ablation Studies**: Understanding component contributions
- **Hyperparameter Tuning**: Training configuration optimization
- **Evaluation Methods**: Proper model assessment techniques

## 🔧 Troubleshooting

### Common Issues

**CUDA Out of Memory:**
```python
# Reduce batch size
physical_batch_size = 16  # Instead of 32
```

**Slow Training:**
```python
# Increase batch size or use more GPUs
effective_batch_size = 128
```

**Poor Translation Quality:**
- Check if tokenizer is properly trained
- Verify dataset quality
- Try different checkpoint steps
- Use beam search instead of greedy decoding

### Performance Tips

1. **Use multiple GPUs**: Significantly faster training
2. **Enable mixed precision**: Reduces memory usage
3. **Monitor GPU utilization**: Ensure efficient resource usage
4. **Save checkpoints frequently**: Avoid losing progress

## 📚 Further Reading

### Essential Papers
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Original Transformer paper
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html) - Detailed implementation guide
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) - Bidirectional Transformers

### Additional Resources
- [Transformer Architecture Explained](https://jalammar.github.io/illustrated-transformer/) - Visual explanations
- [PyTorch Transformer Tutorial](https://pytorch.org/tutorials/beginner/transformer_tutorial.html) - Official tutorial
- [Machine Translation Course](https://web.stanford.edu/class/cs224n/) - Stanford CS224N

## 🤝 Contributing

This is an educational project! Contributions are welcome:

1. **Bug fixes**: Report and fix issues
2. **Documentation**: Improve explanations and examples
3. **Features**: Add new evaluation metrics or training techniques
4. **Optimization**: Improve training speed or memory usage

## 📄 License

This project is for educational purposes. Please cite the original Transformer paper if you use this implementation in your research.

## 🙏 Acknowledgments

- **Original Authors**: Vaswani et al. for the groundbreaking Transformer paper
- **PyTorch Team**: For the excellent deep learning framework
- **Hugging Face**: For tokenization and evaluation tools
- **WMT14 Dataset**: For providing high-quality parallel text data

---

**Happy Learning!** 🚀

This project provides a solid foundation for understanding modern NLP architectures. The hands-on experience of building a Transformer from scratch will give you insights that reading papers alone cannot provide.
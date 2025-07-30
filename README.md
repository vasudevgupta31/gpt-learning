# GPT Research Implementation

A research implementation exploring transformer-based language models, inspired by foundational work in the field including "Attention Is All You Need" and GPT architectures.

## Overview

This project implements a GPT-style transformer model for language modeling research. The codebase includes data preprocessing, model components, training infrastructure, and generation capabilities. While drawing inspiration from breakthrough papers in the field, this remains an experimental research implementation.

## Architecture

The model implements a decoder-only transformer architecture following the GPT paradigm:

- **Multi-head self-attention mechanisms** as described in Vaswani et al. (2017)
- **Positional encodings** for sequence understanding
- **Feed-forward networks** with residual connections
- **Layer normalization** for training stability
- **Causal masking** for autoregressive generation

### Key Components

- `src/components.py`: Core transformer blocks, attention mechanisms, and model architecture
- `src/dataloader.py`: Efficient data loading and batching for large-scale training
- `src/generate.py`: Text generation with various sampling strategies
- `trainer.py`: Training loop with gradient accumulation and optimization
- `preprocessing.py`: Data preprocessing and tokenization utilities

## Datasets

The project includes several curated datasets for training:

- **CNN Articles (2011-2022)**: News articles for diverse language patterns
- **Enron Emails**: Business communication data
- **Project Gutenberg**: Literary works for rich language modeling
- **OpenWebText**: Web-crawled text data
- **Wikipedia**: Encyclopedic content
- **TransWeb Educational**: Academic and educational content

All datasets are preprocessed and sharded for efficient training.

## Configuration

Key hyperparameters (configurable in `trainer.py`):

- **Model size**: 12 layers, 12 attention heads, 768 embedding dimensions
- **Context length**: 512 tokens
- **Vocabulary**: ~50K tokens with byte-pair encoding
- **Batch size**: 8 micro-batches, 262K token total batch size
- **Learning rate**: 6e-4 with cosine decay and warmup
- **Training steps**: 20K steps with gradient accumulation

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd gpt-research-new

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training

```bash
python trainer.py
```

The trainer supports:
- Mixed precision training
- Gradient accumulation
- Learning rate scheduling
- Checkpointing and resumption
- WandB integration for experiment tracking

### Generation

```bash
python src/generate.py
```

Supports various sampling strategies:
- Top-k sampling
- Temperature scaling
- Nucleus (top-p) sampling

### Data Preprocessing

```bash
python preprocessing.py
```

Handles tokenization, dataset chunking, and preprocessing for training.

## Technical Implementation

### Attention Mechanism

Following Vaswani et al., the scaled dot-product attention:

```
Attention(Q,K,V) = softmax(QK^T / √d_k)V
```

### Training Optimizations

- **Weight tying**: Shared embeddings between input and output layers
- **Gradient clipping**: Prevents gradient explosions
- **Weight decay**: L2 regularization for generalization
- **Learning rate warmup**: Stable training initialization

### Memory Efficiency

- **Gradient checkpointing**: Trades compute for memory
- **Mixed precision**: Faster training with FP16
- **Data sharding**: Efficient loading of large datasets

## Experimental Results

This implementation achieves competitive perplexity scores on standard language modeling benchmarks, though specific numbers vary with training configuration and dataset composition.

## Research Context

This work builds upon several key papers:

- **Vaswani et al. (2017)**: "Attention Is All You Need" - Introduced the transformer architecture
- **Radford et al. (2018)**: "Improving Language Understanding by Generative Pre-Training" - GPT-1
- **Radford et al. (2019)**: "Language Models are Unsupervised Multitask Learners" - GPT-2

The implementation attempts to faithfully reproduce core architectural decisions while remaining accessible for research and experimentation.

## Limitations

- **Scale**: This is a research implementation, not optimized for production scale
- **Compute requirements**: Training requires significant GPU resources
- **Dataset size**: Smaller than commercial implementations
- **Evaluation**: Limited evaluation on downstream tasks

## Contributing

This is an experimental research project. Contributions that improve:
- Training efficiency
- Model architecture experiments
- Evaluation methodologies
- Documentation and reproducibility

are welcome.

## License

This project is intended for research and educational purposes. Please respect the licensing terms of the datasets and ensure compliance with relevant usage policies.

## Acknowledgments

We gratefully acknowledge the foundational work of the transformer and GPT research communities. This implementation is built upon decades of research in natural language processing, attention mechanisms, and large-scale neural networks.

Special thanks to Andrej Karpathy, whose educational content, clear explanations, and open-source contributions have been invaluable in deepening our understanding of neural networks and language models. His work has been instrumental in making complex concepts accessible and inspiring this implementation.

Special thanks to the authors of the seminal papers that made this work possible, and to the broader research community for advancing our understanding of language models.

---

*This is a research implementation created for educational and experimental purposes. While we strive for correctness and efficiency, this should not be considered a production-ready system.*
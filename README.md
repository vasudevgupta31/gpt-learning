# GPT Research Implementation

A research implementation exploring transformer-based language models, inspired by foundational work in the field including "Attention Is All You Need" and GPT2/3 papers.

## Overview

This project implements a GPT-style (decoder only) transformer model for language modeling research. The codebase includes data downloading, data preprocessing, model components, training infrastructure, and generation capabilities. While drawing inspiration from papers in the field, this remains an experimental research implementation.

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
- `download_data.py`: Data downloading as batched jsonl files from hugging face

## Datasets


This experiment uses a mix of web, literary, encyclopedic, academic, and formal communication datasets to encourage broad generalization during training. All datasets have been preprocessed and sharded for efficient token loading and training throughput.

| Dataset                      | Description                                                                                       | Size    | Source                                                                                      |
|------------------------------|---------------------------------------------------------------------------------------------------|---------|---------------------------------------------------------------------------------------------|
| **CNN Articles (2011–2022)** | News articles covering over a decade of global events; diverse writing styles and topics.         | 179 MB  | [HF Link](https://huggingface.co/datasets/AyoubChLin/CNN_News_Articles_2011-2022)          |
| **Enron Emails**             | Real-world business communication dataset of emails from the Enron Corporation.                  | 1.1 GB  | [HF Link](https://huggingface.co/datasets/snoop2head/enron_aeslc_emails)                   |
| **Project Gutenberg**        | Digitized public domain books — rich in classical language, style, and long-form structure.      | 36 GB   | [HF Link](https://huggingface.co/datasets/Navanjana/Gutenberg_books)                       |
| **OpenWebText**              | A web-crawled corpus inspired by the original GPT-2 dataset, capturing diverse internet writing. | 38 GB   | [HF Link](https://huggingface.co/datasets/vietgpt/openwebtext_en)                          |
| **Wikipedia (English)**      | Cleaned and structured encyclopedia-style knowledge corpus.                                      | 11 GB   | [HF Link](https://huggingface.co/datasets/lucadiliello/english_wikipedia)                  |
| **TransWeb Edu (English)**   | Educational and academic articles from TransWeb — spanning multiple subjects and exam prep.      | 267 GB  | [HF Link](https://huggingface.co/datasets/britllm/TransWeb-Edu-English)                    |
| **LOTR Books**               | A small curated subset from Tolkien’s *Lord of the Rings* for qualitative inspection.            | 3.2 MB  | _Local extract_                                                                             |


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

This work tries to build upon several foundational papers in neural language modeling and transformer-based architectures:

---
### 1. Vaswani et al. (2017) — *Attention Is All You Need*
[*Attention is All You Need*](https://arxiv.org/abs/1706.03762).

---
### 2. Radford et al. (2019) — *Language Models are Unsupervised Multitask Learners* (GPT-2)
[*Language Models are Unsupervised Multitask Learners*](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf). OpenAI Technical Report.

---
### 3. Brown et al. (2020) — *Language Models are Few-Shot Learners* (GPT-3)
  [*Language Models are Few-Shot Learners*](https://arxiv.org/abs/2005.14165). In *Advances in Neural Information Processing Systems (NeurIPS)*.

---

This implementation replicates core architectural elements of GPT models, optimized for accessibility and experimentation under limited compute.


## Limitations

- **Scale**: This is a research implementation, not optimized for production scale
- **Compute requirements**: Training requires significant GPU resources
- **Dataset size**: Smaller than commercial implementations but about 400GB
- **Evaluation**: Limited evaluation on downstream tasks

## Contributing

This is an experimental research project. Contributions that improve:
- Training efficiency
- Model architecture experiments
- Evaluation methodologies
- Documentation and reproducibility

are welcome.

## License

This project is intended for research and educational purposes.

## Acknowledgments

I gratefully acknowledge the foundational work of the transformer and GPT research communities. This implementation is built upon decades of research in natural language processing, attention mechanisms, and large-scale neural networks.

Special thanks to Andrej Karpathy, whose educational content, clear explanations, and open-source contributions have been invaluable in deepening my understanding of neural networks and language models. His work has been instrumental in making complex concepts accessible and inspiring this implementation.

Special thanks to the authors of the papers, and to the broader research community for advancing the field and our understanding of these models.

---

*This is a research implementation created for educational and experimental purposes. This should not be considered a production-ready system.*

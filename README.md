# Byte Latent Transformer

An experimental implementation exploring entropy-based patching algorithms for byte-level language processing, inspired by Meta's Byte Latent Transformer research.

![Byte Latent Transformer](https://sagarsarkale.com/blt1/000-cover.jpg)

## Overview

This repository implements various byte-level tokenization strategies that use entropy analysis to intelligently segment text at the byte level. Unlike traditional tokenization methods that rely on fixed vocabularies or simple heuristics, this approach uses statistical entropy measures computed from n-gram language models to identify natural boundaries in byte streams.

## Key Concepts

### 1. Byte-Level Processing
Text is converted directly to UTF-8 byte sequences, allowing the model to handle any character or language without pre-defined vocabularies.

![Byte-based System](https://sagarsarkale.com/blt1/001-byte-based.png)

Traditional tokenizers require vocabulary training and preprocessing, while byte-based systems work directly with UTF-8 bytes.

### 2. Entropy-Based Patching
Multiple patching strategies are implemented:
- **Fixed-Length Patching**: Splits text into fixed-size byte chunks
- **Space-Based Patching**: Uses whitespace as natural boundaries
- **Global Threshold Patching**: Creates patches based on entropy exceeding a global threshold ($\mu$ + $\sigma$)
- **Monotonic Constraint Patching**: Segments text where entropy shows significant drops
- **Double Derivative Patching**: Uses second-order entropy changes to detect patch boundaries

![Fixed Patching](https://sagarsarkale.com/blt1/003-fixed-patching.png)
*Fixed K-byte patching with consistent intervals*

![Space Patching](https://sagarsarkale.com/blt1/004-space-patching.png)
*Space-based patching at whitespace boundaries*

![Entropy Patching](https://sagarsarkale.com/blt1/005-entropy-diagram-1.0.png)
*Entropy values across a sentence showing natural patch boundaries*

### 3. N-gram Language Models
Builds statistical models (n=2,3,4,5) from a corpus of classic literature to estimate byte-level entropy and predictability.

## Project Structure

```
byte_latent_transformer/
└── 001/
    ├── 00_chars2bytes.py          # Character to byte conversion basics
    ├── 01_types_of_patching.py    # Fixed-length and space-based patching
    ├── 02_entropy_example.py      # Entropy calculation with 2-gram models
    ├── 03_entropy_patching.py     # Combined n-gram (3,4,5) entropy patching
    └── 04_expt_patching.py        # Experimental patching with visualization
```

## Getting Started

### Prerequisites

```bash
pip install -r byte_latent_transformer/requirements.txt
```

### Running the Code

The code is structured as interactive Python scripts with Jupyter cell markers (`#%%`). You can run them in:
- VS Code with Python extension
- Cursor with interactive mode
- Jupyter notebooks
- Any IDE supporting cell execution

### Quick Start

1. **Basic Byte Conversion**:
```python
python byte_latent_transformer/001/00_chars2bytes.py
```

2. **Compare Patching Methods**:
```python
python byte_latent_transformer/001/01_types_of_patching.py
```

3. **Entropy Analysis**:
```python
python byte_latent_transformer/001/02_entropy_example.py
```

## How It Works

### Entropy Calculation

For each byte position, the algorithm:
1. Extracts the n-gram context (preceding n bytes)
2. Looks up the context in pre-built frequency models
3. Calculates Shannon entropy: $-\sum(p \times log_2(p))$
4. Uses entropy to determine patch boundaries

### Training Corpus

The implementation uses classic literature from Project Gutenberg:
- Sherlock Holmes
- Pride and Prejudice
- Alice in Wonderland
- Frankenstein

Models are cached in `./models/` directory for fast re-use.

## Features

- **Multiple N-gram Models**: Combines 2-gram, 3-gram, 4-gram, and 5-gram models
- **Parallel Processing**: ThreadPoolExecutor for fast n-gram computation
- **Model Caching**: Pickle-based caching to avoid rebuilding models
- **Visualization**: Matplotlib plots showing entropy patterns and patch boundaries
- **Comparative Analysis**: Side-by-side comparison of different patching strategies

## Example Output

```
Analyzing: I walked to the store and bought a book.

Combined N-gram Global Threshold Patches:
+-------+-----+--------+----+-------+-----+--------+-----+--------+
| Patch | 1   | 2      | 3  | 4     | 5   | 6      | 7   | 8      |
+-------+-----+--------+----+-------+-----+--------+-----+--------+
| Input | I   | walked | to | the   | ... | bought | a   | book.  |
| #Bytes| 1   | 7      | 3  | 4     | ... | 7      | 2   | 5      |
+-------+-----+--------+----+-------+-----+--------+-----+--------+
```

![Patching Comparison](https://sagarsarkale.com/blt1/008-patches-1.png)
*Comparison of global threshold and monotonic constraint patching methods*

![Entropy Analysis](https://sagarsarkale.com/blt1/006-entropy-diagram-2.png)
*Entropy spikes occur at unusual words and symbols, identifying natural boundaries*

## Environment Variables

- `MODEL_DIR`: Directory for storing n-gram models (default: `./models`)

## Blog & Research

This implementation supports research and blog posts on byte-level language processing.

Blog: https://sagarsarkale.com/blog

## Maintainer

[Sagar Sarkale](https://www.linkedin.com/in/sagar-sarkale)

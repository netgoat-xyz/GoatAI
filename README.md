# Koda

A lightweight, high-performance tabular classifier for real-time DDoS attack detection, delivering accurate results in seconds.

## Features

- **Fast**: Detection in under a few seconds
- **Compact**: Only 24.93 KB total parameters
- **Accurate**: 98.54% accuracy on synthetic benchmarks (v1)
- **Easy to use**: Simple Python API with minimal dependencies

## Resources

All model artifacts are available on Hugging Face under the [netgoat-ai](https://huggingface.co/netgoat-ai) organization:

- **[Model Weights](https://huggingface.co/netgoat-ai/koda-2)** - Pre-trained model ready for inference
- **[Demo Space](https://huggingface.co/spaces/netgoat-ai/koda-2-space)** - Interactive web demo
- **[Dataset](https://huggingface.co/datasets/netgoat-ai/SynthDDoS)** - Synthetic DDoS training data

## Requirements

### Python Version
- Python 3.8 or higher

### Dependencies

```bash
pip install pandas numpy tensorflow scikit-learn
```

Or install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

**requirements.txt:**
```
pandas>=1.3.0
numpy>=1.21.0
tensorflow>=2.10.0
scikit-learn>=1.0.0
```

### Library Breakdown

| Library | Purpose |
|---------|---------|
| `pandas` | Data manipulation and CSV handling |
| `numpy` | Numerical operations and array processing |
| `tensorflow` | Deep learning framework for model training |
| `scikit-learn` | Data preprocessing (MinMaxScaler) |

## Repository Structure

The `src` directory contains the dataset generator and two versioned trainers:

1. **`make_dataset.py`** - Generates the synthetic DDoS dataset
2. **`koda-1/train.py`** - Trains the first Koda model on the generated dataset
3. **`koda-2/train.py`** - Trains the second Koda model on the generated dataset and CIC-DDOS2019

## Model Specifications

| Metric | Value |
|--------|-------|
| Total parameters | 6,382 (24.93 KB) |
| Trainable parameters | 5,982 (23.37 KB) |
| Non-trainable parameters | 400 (1.56 KB) |

## Benchmark Results for v1 (Synthetic Dataset)

| Metric | Score |
|--------|-------|
| Accuracy | 98.54% |
| Precision | 97.16% |
| Recall | 100.00% |

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/netgoat-xyz/koda.git
cd koda
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Training from Scratch

```bash
# Generate synthetic dataset
python src/make_dataset.py

# Train the model
python src/koda-1/train.py # uses dataset.csv from the repository root

# or
python src/koda-2/train.py # requires CIC-DDOS2019 files in the repository root.
# replace the path to the synthetic dataset where you saved the dataset that you generate

```

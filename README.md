# GoatAI

A lightweight, high-performance tabular classifier for real-time DDoS attack detection, delivering accurate results in seconds.

## Features

- **Fast**: Detection in under a few seconds
- **Compact**: Only 24.93 KB total parameters
- **Accurate**: 98.54% accuracy on synthetic benchmarks
- **Easy to use**: Simple Python API with minimal dependencies

## Resources

All model artifacts are available on Hugging Face under the [netgoat-ai](https://huggingface.co/netgoat-ai) organization:

- **[Model Weights](https://huggingface.co/netgoat-ai/GoatAI)** - Pre-trained model ready for inference
- **[Demo Space](https://huggingface.co/spaces/netgoat-ai/GoatAI-space)** - Interactive web demo
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

The `src` directory contains two main Python scripts:

1. **`make_dataset.py`** - Generates the synthetic DDoS dataset
2. **`train.py`** - Trains the model on the generated dataset

## Model Specifications

| Metric | Value |
|--------|-------|
| Total parameters | 6,382 (24.93 KB) |
| Trainable parameters | 5,982 (23.37 KB) |
| Non-trainable parameters | 400 (1.56 KB) |

## Benchmark Results (Synthetic Dataset)

| Metric | Score |
|--------|-------|
| Accuracy | 98.54% |
| Precision | 97.16% |
| Recall | 100.00% |

## Getting Started

### Installation

1. Clone the repository:
```bash
git clone https://github.com/netgoat-xyz/GoatAI.git
cd GoatAI
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
python src/train.py
```

![NetGoat Asset](./asset/banner.png)

# Koda

A collection of compact traffic-security models for DDoS detection and HTTP request inspection.

## Features

- **Fast**: Koda-3 performs about 345,000 single-flow predictions per second in the local benchmark
- **Compact**: The tested Koda-3 binary artifact is 483 bytes
- **Measured honestly**: Same-distribution validation and a harder frozen challenge set are reported separately
- **Deployable**: Koda-3 and Koda-WAF use only the Python standard library at runtime
- **Learned HTTP inspection**: Koda-WAF classifies decoded request features without a benign allowlist

## Resources

All model artifacts are available on Hugging Face under the [netgoat-ai](https://huggingface.co/netgoat-ai) organization:

- **[Model Weights](https://huggingface.co/netgoat-ai/koda-2)** - Pre-trained model ready for inference
- **[Demo Space](https://huggingface.co/spaces/netgoat-ai/koda-2-space)** - Interactive web demo
- **[Dataset](https://huggingface.co/datasets/netgoat-ai/SynthDDoS)** - Synthetic DDoS training data

## Requirements

### Python Version
- Python 3.10 or higher

### Dependencies

Koda-3 and Koda-WAF require no third-party Python packages. The legacy Koda-1 and Koda-2 trainers use:

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

The `src` directory contains the dataset generator, versioned DDoS models, and the HTTP WAF:

1. **`make_dataset.py`** - Generates the synthetic DDoS dataset
2. **`koda-1/train.py`** - Trains the first Koda model on the generated dataset
3. **`koda-2/train.py`** - Trains the second Koda model on the generated dataset and CIC-DDOS2019
4. **`koda-3/`** - Streams numeric CSV data into a compact supervised Gaussian classifier
5. **`koda-waf/`** - Runs a dependency-free HTTP request inspection service

Both ready-to-run binary artifacts are generated or downloaded into one ignored folder:

```text
models/
├── koda-3.pkl
└── koda-waf.pkl
```

The `.pkl` files contain learned model parameters, not JSON configuration. Only load model files from trusted sources because Python pickle files can execute code while loading.

Model binaries and generated CSV datasets are intentionally excluded from Git history. Download both model assets from the GitHub release into `models/`, or retrain them with the commands below. The tracked `models/.gitkeep` preserves the destination directory.

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

## Koda-3 benchmark

The local benchmark generates 100,000 rows using the same distributions as `make_dataset.py`, reserves a deterministic 20% validation split, and performs 100,000 single-row predictions. A separate frozen 10,000-row challenge set shifts the distributions and adds bursty benign traffic, smaller SYN floods, and low-rate attacks.

| Metric | Result |
|--------|--------|
| Same-distribution validation accuracy | 100.00% |
| Dedicated challenge accuracy | 68.37% |
| Challenge precision / recall | 100.00% / 36.74% |
| Streaming training time | 1.07 seconds |
| Inference throughput | 345,416 predictions/second |
| Model artifact size | 483 bytes |

The perfect validation score is caused by simple, strongly separated synthetic classes—not proof of real-world accuracy. The challenge result shows that flood-only training misses many smaller and low-rate attacks. Times were measured on the current development machine with Python 3.14. See [the test report](docs/TEST_RESULTS.md) for details.

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

# Train Koda-3 in one streaming pass
python src/koda-3/train.py --data dataset.csv --test-data data/koda-3-test.csv

# Reproduce the frozen challenge set
python src/koda-3/make_test_set.py

# Classify one flow using a JSON object
python src/koda-3/infer.py --input '{"Flow Duration":5000,"Total Fwd Packets":1500,"Total Backward Packets":1,"Packet Length Mean":1200,"Flow IAT Mean":50,"Fwd Flag Count":1}'
```

### Koda-WAF

```bash
python src/koda-waf/server.py --port 8090

# Retrain from a labeled request CSV
python src/koda-waf/train.py --data /path/to/waf_train_augmented.csv

# Evaluate the binary model on the frozen project test set
python src/koda-waf/evaluate.py --model models/koda-waf.pkl
```

Benign requests return HTTP 200 and detected attacks return HTTP 403. The engine recursively decodes URL, HTML, and Base64 variants, extracts token, phrase, context, and signature features, and passes those features to the learned binary classifier. Signatures provide features and explanations but do not automatically block requests.

The current model scores 99.29% on its training validation split, 95.83% on the frozen project test set, and 95.06% in GoTestWAF. See [the test report](docs/TEST_RESULTS.md) for benchmark limitations.

### Tests

```bash
python -m unittest discover -s src/koda-3 -p 'test_*.py' -v
python -m unittest discover -s src/koda-waf -p 'test_*.py' -v
python src/koda-waf/evaluate.py
```

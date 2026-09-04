# Koda-3 and Koda-WAF

This release contains two dependency-free Python model artifacts:

- `koda-3.pkl` — compact supervised Gaussian DDoS classifier.
- `koda-waf.pkl` — learned HTTP request classifier with 60,000 weighted features and no benign allowlist.

## Verified results

### Koda-3

- Same-distribution validation accuracy: 100.00%
- Distribution-shifted challenge accuracy: 68.37%
- Model size: 483 bytes

### Koda-WAF

- Training validation accuracy: 99.29%
- Frozen project test accuracy: 95.83%
- GoTestWAF score: 95.06%
- GoTestWAF attacks blocked: 608/673 resolved cases
- GoTestWAF benign cases allowed: 127/141
- Model size: 2,309,887 bytes

## Installation

Download both files into the repository's `models/` directory. Only load trusted pickle files.

```bash
mkdir -p models
python src/koda-3/infer.py --model models/koda-3.pkl --input '{"Flow Duration":5000,"Total Fwd Packets":1500,"Total Backward Packets":1,"Packet Length Mean":1200,"Flow IAT Mean":50,"Fwd Flag Count":1}'
python src/koda-waf/server.py --model models/koda-waf.pkl --port 8090
```

## SHA-256

```text
ccc3af877fe51a27c729aa3837604bbf21a4119241c2e1fd8e5d914ab835d95d  koda-3.pkl
9852780c484b358c3a297aee1aa0ad825b015fb2d4199bf8b792f3939ac8a642  koda-waf.pkl
```


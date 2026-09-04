# Koda-3 and Koda-WAF test results

Tested locally on 2026-09-03 with Python 3.14.7 and Go 1.27.0.

## Koda-3 same-distribution validation

The benchmark generated 100,000 deterministic synthetic rows from the same feature distributions used by `src/make_dataset.py`. It trained on 80,159 rows and evaluated an untouched deterministic holdout of 19,841 rows.

| Metric | Result |
|---|---:|
| Accuracy | 100.00% |
| Precision | 100.00% |
| Recall | 100.00% |
| Specificity | 100.00% |
| Training time | 1.022 s |
| Artifact size | 483 bytes |
| Inference throughput | 345,416 predictions/s |
| Mean inference latency | 2.895 microseconds/flow |

This validates implementation speed and confirms that the classifier learns the original synthetic generator. It does not establish real-world accuracy because the training and validation rows come from the same strongly separated distributions.

## Koda-3 dedicated challenge set

`data/koda-3-test.csv` is a frozen, independently seeded 10,000-row test set. It is never used for fitting, variance estimation, or threshold selection. It shifts feature distributions and includes ordinary and bursty benign traffic plus volumetric, smaller SYN-flood, and low-rate attack variants.

SHA-256: `abd47bfc76abe82c9dd3f09679f4f08fee8aaeebe6f589428d2559a60cd89919`

| Metric | Result |
|---|---:|
| Rows | 10,000 |
| Accuracy | 68.37% |
| Precision | 100.00% |
| Recall | 36.74% |
| Specificity | 100.00% |
| True positives | 1,837 |
| True negatives | 5,000 |
| False positives | 0 |
| False negatives | 3,163 |

The result explains the earlier perfect number: the original benign and flood classes are almost trivially separable. Under distribution shift, Koda-3 remains conservative but misses many attacks that do not resemble the original high-volume flood generator. This dedicated set is now frozen; future model changes should not tune thresholds or features against it. A new untouched test version is needed after substantial iteration.

## Koda-WAF

Koda-WAF was trained on 166,454 rows from the augmented request corpus, with 18,374 rows reserved for threshold selection and validation. It learned 60,000 weighted token, phrase, context, and signature features. Labeled hard-negative rows receive extra training weight; there is no runtime benign allowlist.

| Training metric | Result |
|---|---:|
| Validation accuracy | 99.28% |
| Validation precision | 99.71% |
| Validation recall | 98.58% |
| Validation specificity | 99.79% |
| Training time | 46.11 s |
| Artifact size | 2,309,887 bytes |

GoTestWAF was built from the vendored source at `research/netgoat-ai/koda-waf-2/.tools/gotestwaf-src` and run against the learned `models/koda-waf.pkl` artifact with 20 workers and `--skipWAFBlockCheck`.

| Metric | Result |
|---|---:|
| Overall score | 95.06% |
| Resolved attacks blocked | 608 / 673 (90.34%) |
| Benign tests allowed | 127 / 141 (90.07%) |
| Unresolved tests | 2 / 675 |
| Failed requests | 0 |

The two unresolved cases are GoTestWAF API checks unavailable to this HTTP-only target. The complete generated JSON report for this run is in the ignored `reports/gotestwaf/learned-model-v4/` directory.

The augmented training corpus contains 282 labeled GoTestWAF-style false-positive examples, so this is not a fully independent generalization benchmark. It is still more representative than the previous allowlist-backed result because every decision now comes from learned feature weights and one validation-selected threshold. Novel attacks and production traffic require separate testing.

## Koda-WAF project test set

`data/koda-waf-test.jsonl` contains 24 frozen request views that are separate from GoTestWAF: 12 benign application/documentation requests and 12 malicious requests across the supported attack families.

SHA-256: `ce25fab8b43e51ec84af8f4765095c2a94bf9b98d4076f68a116c4c498c67828`

| Metric | Result |
|---|---:|
| Accuracy | 95.83% |
| Precision | 100.00% |
| Recall | 91.67% |
| Specificity | 100.00% |

The model allowed all 12 benign cases and blocked 11 of 12 attacks. It missed one short command-injection request. This remains a small, author-created functional corpus and should be expanded with sanitized production traffic and independently collected bypass samples.

## Commands

```bash
python -m unittest discover -s src/koda-3 -p 'test_*.py' -v
python -m unittest discover -s src/koda-waf -p 'test_*.py' -v
python src/koda-3/make_test_set.py
python src/koda-3/train.py --data dataset.csv --test-data data/koda-3-test.csv
python src/koda-waf/evaluate.py

go build -o /tmp/koda-gotestwaf ./cmd/gotestwaf
python src/koda-waf/server.py --port 8093
/tmp/koda-gotestwaf \
  --url=http://127.0.0.1:8093 \
  --noEmailReport --workers=20 \
  --reportPath=reports/gotestwaf/latest --reportFormat=json \
  --configPath=/path/to/gotestwaf/config.yaml \
  --testCasesPath=/path/to/gotestwaf/testcases \
  --skipWAFBlockCheck
```

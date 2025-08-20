# GoT for Bias Mitigation — Benchmark Report

- Date: 20250820_174033
- Dataset: `synthetic`
- Samples: 50
- GoT params: max_depth=5, bias_threshold=0.15

## Aggregate Metrics (means)

| Variant | mean_cbs ↓ | mean_crs | mean_sim | mean_replace_rate | mean_time_sec | n |
|---|---:|---:|---:|---:|---:|---:|
| GoT | 0.1411 | 0.9975 | 0.9937 | 0.0013 | 0.0233 | 50 |
| fixed_order_mask | 0.2334 | 0.6215 | 0.5032 | 0.1808 | 0.0000 | 50 |
| identity | 0.1370 | 1.0000 | 1.0000 | 0.0000 | 0.0000 | 50 |
| privacy_only | 0.1643 | 0.9240 | 0.9151 | 0.0234 | 0.0000 | 50 |
| random_mask | 0.1936 | 0.8844 | 0.7716 | 0.0000 | 0.0000 | 50 |
| stereotypes_then_privacy | 0.2157 | 0.7135 | 0.6396 | 0.1250 | 0.0000 | 50 |

Notes:
- `cbs` (Composite Bias Score) is lower-better, computed via got.py weights.
- `crs` (Content Retention Score) ~ SequenceMatcher ratio vs original (0–1).
- `sim` is SBERT cosine if available, else SequenceMatcher.
- `replace_rate` is fraction of `[MASK:...]` tokens among words (approx.).
- Expect GoT to achieve **lower mean_cbs** than fixed-order baselines while keeping **crs/sim** high.

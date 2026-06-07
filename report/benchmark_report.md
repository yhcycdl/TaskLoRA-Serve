# Benchmark Report

## Workload

The V1 mixed workload uses:

- 50% code generation
- 40% math reasoning
- 10% general chat

## Commands

```bash
python -m benchmark.loadgen \
  --config configs/benchmark.yaml \
  --url http://localhost:8000/v1/task/chat \
  --concurrency 4 \
  --duration 60 \
  --output benchmark/results/v1_c4.jsonl

python -m benchmark.analyze_results benchmark/results/v1_c4.jsonl
```

## Results To Fill After Running

| Concurrency | RPS | Tokens/s | p50 ms | p95 ms | p99 ms | Error Rate |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | TBD | TBD | TBD | TBD | TBD | TBD |
| 4 | TBD | TBD | TBD | TBD | TBD | TBD |
| 8 | TBD | TBD | TBD | TBD | TBD | TBD |

## Adapter Distribution

| Adapter | Requests | Share |
| --- | ---: | ---: |
| code-lora | TBD | TBD |
| math-lora | TBD | TBD |
| base | TBD | TBD |


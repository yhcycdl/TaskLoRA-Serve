# Evaluation Report

## Metrics

- GSM8K: final-answer exact match.
- MBPP: pass@1 using `test_list` execution with timeout.

## Commands

```bash
python -m evaluation.eval_base_vs_lora --config configs/eval.yaml
```

## Results To Fill After Running

| Task | Model | Samples | Score |
| --- | --- | ---: | ---: |
| GSM8K | base | 200 | TBD |
| GSM8K | math-lora | 200 | TBD |
| MBPP | base | 50 | TBD |
| MBPP | code-lora | 50 | TBD |

## Failure Analysis Template

- Cases where LoRA improves: TBD
- Cases where LoRA does not improve: TBD
- Common math failure type: TBD
- Common code failure type: TBD

If a task-specific adapter improves its target task but not the other task, report that as a useful trade-off rather than hiding it.


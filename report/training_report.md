# Training Report

## Configuration

- Base model: `Qwen/Qwen2.5-1.5B-Instruct`
- Code data: `sahil2801/CodeAlpaca-20k`
- Math data: `openai/gsm8k`
- Method: QLoRA
- LoRA rank: 16
- LoRA alpha: 32
- LoRA dropout: 0.05
- Max sequence length: 2048
- Batch size: 1
- Gradient accumulation: 8

## Commands

```bash
python -m training.build_dataset --task all --output-dir data/processed
python -m training.train_qlora --config configs/code_lora.yaml
python -m training.train_qlora --config configs/math_lora.yaml
```

## Results To Fill After Running

| Adapter | Samples | Epochs | Train Loss | Eval Loss | Peak VRAM | Runtime |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| code-lora | TBD | 2 | TBD | TBD | TBD | TBD |
| math-lora | TBD | 2 | TBD | TBD | TBD | TBD |

## Notes

Run smoke training first with `--max-train-samples 10 --max-eval-samples 5 --max-steps 1` to validate the environment before paying for a full GPU run.


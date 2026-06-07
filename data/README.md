# Data

TaskLoRA-Serve V1 uses public Hugging Face datasets:

- Code SFT: `sahil2801/CodeAlpaca-20k`
- Math SFT/eval: `openai/gsm8k`
- Code eval: `google-research-datasets/mbpp`

Build processed chat-format JSONL files with:

```bash
python -m training.build_dataset --task all --output-dir data/processed
```

The repository does not commit downloaded or processed datasets.


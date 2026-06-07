from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

from training.train_utils import ensure_dir, format_chat_messages, load_yaml


def resolve_torch_dtype(torch_module: Any, dtype_name: str | None):
    if not dtype_name:
        return None
    mapping = {
        "float16": torch_module.float16,
        "fp16": torch_module.float16,
        "bfloat16": torch_module.bfloat16,
        "bf16": torch_module.bfloat16,
        "float32": torch_module.float32,
        "fp32": torch_module.float32,
    }
    return mapping.get(str(dtype_name).lower())


def resolve_device_map(torch_module: Any, use_qlora: bool):
    if not use_qlora:
        return "auto"
    if torch_module.cuda.is_available():
        return {"": torch_module.cuda.current_device()}
    if hasattr(torch_module, "xpu") and torch_module.xpu.is_available():
        return {"": torch_module.xpu.current_device()}
    raise RuntimeError(
        "QLoRA 4-bit training requires an available GPU/XPU. "
        "Check your PyTorch CUDA installation with `torch.cuda.is_available()`."
    )


def validate_single_gpu_qlora(torch_module: Any, use_qlora: bool) -> None:
    if not use_qlora or not torch_module.cuda.is_available():
        return
    visible_gpu_count = torch_module.cuda.device_count()
    if visible_gpu_count > 1:
        raise RuntimeError(
            "TaskLoRA-Serve V1 uses single-GPU QLoRA training. "
            f"Your process can see {visible_gpu_count} GPUs, so Transformers may wrap the "
            "4-bit model with DataParallel and fail. Re-run with one visible GPU, for example: "
            "`CUDA_VISIBLE_DEVICES=0 python -m training.train_qlora ...`"
        )


def training_args_kwargs(training_args_cls: Any, config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    signature = inspect.signature(training_args_cls.__init__)
    params = signature.parameters
    kwargs: dict[str, Any] = {
        "output_dir": output_dir,
        "per_device_train_batch_size": config.get("per_device_train_batch_size", 1),
        "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 8),
        "learning_rate": float(config.get("learning_rate", 2.0e-4)),
        "num_train_epochs": float(config.get("num_train_epochs", 2)),
        "warmup_ratio": float(config.get("warmup_ratio", 0.03)),
        "weight_decay": float(config.get("weight_decay", 0.01)),
        "logging_steps": int(config.get("logging_steps", 10)),
        "save_steps": int(config.get("save_steps", 250)),
        "eval_steps": int(config.get("eval_steps", 100)),
        "save_total_limit": 2,
        "gradient_checkpointing": True,
        "report_to": [],
        "seed": int(config.get("seed", 42)),
    }

    if config.get("max_steps") is not None:
        kwargs["max_steps"] = int(config["max_steps"])

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "steps"
    else:
        kwargs["evaluation_strategy"] = "steps"

    if "bf16" in params:
        kwargs["bf16"] = True
    if "fp16" in params:
        kwargs["fp16"] = False

    return kwargs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a QLoRA adapter from a JSONL chat dataset.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--no-qlora", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(args.config)
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps

    try:
        import torch
        from datasets import load_dataset
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Training dependencies are missing. Install them with "
            "`pip install -r requirements.txt` in a Python 3.10/3.11 environment."
        ) from exc

    output_dir = str(config["output_dir"])
    ensure_dir(output_dir)

    tokenizer = AutoTokenizer.from_pretrained(config["base_model"], trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    qlora_cfg = config.get("qlora", {})
    quantization_config = None
    use_qlora = bool(qlora_cfg.get("enabled", True)) and not args.no_qlora
    validate_single_gpu_qlora(torch, use_qlora)
    if use_qlora:
        compute_dtype = resolve_torch_dtype(torch, qlora_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=bool(qlora_cfg.get("load_in_4bit", True)),
            bnb_4bit_quant_type=qlora_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
        )

    model = AutoModelForCausalLM.from_pretrained(
        config["base_model"],
        quantization_config=quantization_config,
        device_map=resolve_device_map(torch, use_qlora),
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if use_qlora:
        model = prepare_model_for_kbit_training(model)

    lora = config.get("lora", {})
    peft_config = LoraConfig(
        r=int(lora.get("r", 16)),
        lora_alpha=int(lora.get("alpha", 32)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", [])),
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    data_files = {"train": config["train_file"], "validation": config["valid_file"]}
    dataset = load_dataset("json", data_files=data_files)
    if args.max_train_samples is not None:
        dataset["train"] = dataset["train"].select(range(min(args.max_train_samples, len(dataset["train"]))))
    if args.max_eval_samples is not None:
        dataset["validation"] = dataset["validation"].select(
            range(min(args.max_eval_samples, len(dataset["validation"])))
        )

    max_seq_length = int(config.get("max_seq_length", 2048))

    def tokenize_batch(batch: dict[str, Any]) -> dict[str, Any]:
        texts = [format_chat_messages(messages, tokenizer) for messages in batch["messages"]]
        return tokenizer(texts, truncation=True, max_length=max_seq_length, padding=False)

    tokenized = dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=dataset["train"].column_names,
        desc="Tokenizing chat records",
    )

    training_args = TrainingArguments(
        **training_args_kwargs(TrainingArguments, config, output_dir)
    )
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=collator,
    )

    result = trainer.train()
    metrics = result.metrics
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    trainer.save_state()

    log = {
        "run_name": config.get("run_name"),
        "base_model": config["base_model"],
        "train_file": config["train_file"],
        "valid_file": config["valid_file"],
        "output_dir": output_dir,
        "metrics": metrics,
        "qlora_enabled": use_qlora,
    }
    with Path(output_dir, "train_log.json").open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    print(json.dumps(log, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

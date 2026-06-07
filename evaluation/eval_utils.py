from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from training.train_utils import ensure_dir, format_prompt_messages, load_yaml, write_jsonl

NUMBER_RE = re.compile(r"-?\d+(?:,\d{3})*(?:\.\d+)?")


def normalize_number(text: str | None) -> str | None:
    if text is None:
        return None
    return text.replace(",", "").strip()


def extract_gsm8k_answer(text: str) -> str | None:
    marker = "####"
    if marker in text:
        tail = text.split(marker)[-1]
        match = NUMBER_RE.search(tail)
        return normalize_number(match.group(0)) if match else None

    matches = NUMBER_RE.findall(text)
    if not matches:
        return None
    return normalize_number(matches[-1])


def exact_match(prediction: str | None, target: str | None) -> bool:
    return prediction is not None and target is not None and prediction == target


def extract_code_block(text: str) -> str:
    match = re.search(r"```(?:python)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text.strip()


def load_model_and_tokenizer(
    model_id: str,
    adapter_path: str | None = None,
    load_in_4bit: bool = False,
):
    try:
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    except ImportError as exc:
        raise RuntimeError(
            "Evaluation dependencies are missing. Install them with "
            "`pip install -r requirements.txt` in a Python 3.10/3.11 environment."
        ) from exc

    quantization_config = None
    if load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        quantization_config=quantization_config,
        trust_remote_code=True,
    )
    if adapter_path:
        model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()
    return model, tokenizer


def generate_response(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int,
    temperature: float,
) -> str:
    import torch

    prompt = format_prompt_messages(messages, tokenizer)
    encoded = tokenizer(prompt, return_tensors="pt").to(model.device)
    do_sample = temperature > 0
    generate_kwargs: dict[str, Any] = {
        **encoded,
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generate_kwargs["temperature"] = temperature
    with torch.no_grad():
        output_ids = model.generate(**generate_kwargs)
    generated = output_ids[0][encoded["input_ids"].shape[-1] :]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


__all__ = [
    "exact_match",
    "extract_code_block",
    "extract_gsm8k_answer",
    "generate_response",
    "load_model_and_tokenizer",
    "load_yaml",
    "write_jsonl",
    "write_summary",
]

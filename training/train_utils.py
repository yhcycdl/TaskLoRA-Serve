from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> int:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with target.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            count += 1
    return count


def validate_messages(record: dict[str, Any]) -> bool:
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for message in messages:
        if not isinstance(message, dict):
            return False
        if message.get("role") not in {"system", "user", "assistant"}:
            return False
        if not isinstance(message.get("content"), str) or not message["content"].strip():
            return False
    return True


def format_chat_messages(messages: list[dict[str, str]], tokenizer: Any | None = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    rendered: list[str] = []
    for message in messages:
        rendered.append(f"<|{message['role']}|>\n{message['content']}")
    rendered.append("<|end|>")
    return "\n".join(rendered)


def format_prompt_messages(messages: list[dict[str, str]], tokenizer: Any | None = None) -> str:
    if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return "\n".join(f"{m['role']}: {m['content']}" for m in messages) + "\nassistant:"


def truncate_text(text: str, max_chars: int = 8000) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rstrip()


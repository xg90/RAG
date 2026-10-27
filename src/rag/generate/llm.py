"""Unified interface to load and run a causal language model (transformers).

The actual model deployment and invocation happen in notebooks/03_deploy_generate.ipynb
(Kaggle GPU); this module only wraps load_model() / generate() so the heavy
dependencies (torch, transformers) are imported lazily.
"""

from __future__ import annotations

from typing import Any

from .. import config


def load_model(
    model_name: str | None = None,
    device: str = "auto",
    load_in_4bit: bool = False,
) -> tuple[Any, Any]:
    """Load a causal LM and its tokenizer; returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    model_name = model_name or config.LLM_MODEL
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    kwargs: dict[str, Any] = {"device_map": device}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    return model, tokenizer


def generate(
    model: Any,
    tokenizer: Any,
    messages: list[dict[str, str]],
    max_new_tokens: int = 256,
    temperature: float = 0.0,
) -> str:
    """Generate a response for a chat-style messages list; returns decoded text."""
    import torch

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=temperature > 0,
            temperature=temperature,
            pad_token_id=tokenizer.eos_token_id,
        )
    start = inputs["input_ids"].shape[1]
    decoded = tokenizer.decode(outputs[0][start:], skip_special_tokens=True)
    return decoded.strip()


def generate_deepseek(
    messages: list[dict[str, str]],
    model: str | None = None,
    api_key: str | None = None,
    base_url: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
) -> str:
    """Generate a response via the DeepSeek API (OpenAI-compatible); returns text.

    Reads the API key / base URL / model from config unless overridden. The
    `openai` SDK is imported lazily so retrieval-only runs don't require it.
    """
    from openai import OpenAI

    client = OpenAI(
        api_key=api_key or config.DEEPSEEK_API_KEY,
        base_url=base_url or config.DEEPSEEK_BASE_URL,
    )
    resp = client.chat.completions.create(
        model=model or config.DEEPSEEK_MODEL,
        messages=messages,
        temperature=config.DEEPSEEK_TEMPERATURE if temperature is None else temperature,
        max_tokens=config.DEEPSEEK_MAX_TOKENS if max_tokens is None else max_tokens,
    )
    return (resp.choices[0].message.content or "").strip()

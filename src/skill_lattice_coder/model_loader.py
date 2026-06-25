from pathlib import Path

from .config import base_config


def load_model(adapter: Path | None = None, model_name: str | None = None):
    from mlx_lm import load

    return load(
        model_name or base_config()["model"],
        adapter_path=str(adapter) if adapter else None,
    )


def generate_text(
    model,
    tokenizer,
    prompt: str | None = None,
    *,
    messages: list[dict[str, str]] | None = None,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> tuple[str, int, int]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    config = base_config()
    if prompt is not None and messages is not None:
        raise ValueError("provide either prompt or messages, not both")
    if prompt is None and not messages:
        raise ValueError("prompt or messages is required")
    resolved_messages = messages or [{"role": "user", "content": prompt or ""}]
    formatted = tokenizer.apply_chat_template(
        resolved_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenizer.eos_token_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    output = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=max_tokens or config["max_tokens"],
        sampler=make_sampler(config["temperature"] if temperature is None else temperature),
        verbose=False,
    )
    output = output.split("<|im_end|>", 1)[0].rstrip()
    return output, len(tokenizer.encode(formatted)), len(tokenizer.encode(output))

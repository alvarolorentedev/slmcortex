from pathlib import Path

from .config import base_config


def load_model(adapter: Path | None = None):
    from mlx_lm import load

    return load(
        base_config()["model"],
        adapter_path=str(adapter) if adapter else None,
    )


def generate_text(model, tokenizer, prompt: str) -> tuple[str, int, int]:
    from mlx_lm import generate
    from mlx_lm.sample_utils import make_sampler

    config = base_config()
    formatted = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    tokenizer.eos_token_ids.add(tokenizer.convert_tokens_to_ids("<|im_end|>"))
    output = generate(
        model,
        tokenizer,
        prompt=formatted,
        max_tokens=config["max_tokens"],
        sampler=make_sampler(config["temperature"]),
        verbose=False,
    )
    output = output.split("<|im_end|>", 1)[0].rstrip()
    return output, len(tokenizer.encode(formatted)), len(tokenizer.encode(output))

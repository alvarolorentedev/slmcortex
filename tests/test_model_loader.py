import mlx_lm
import mlx_lm.sample_utils

from skill_lattice_coder.model_loader import generate_text


class FakeTokenizer:
    eos_token_ids = {1}

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert messages == [{"role": "user", "content": "Fix it"}]
        assert not tokenize
        assert add_generation_prompt
        return "formatted prompt"

    def encode(self, text):
        return text.split()

    def convert_tokens_to_ids(self, token):
        assert token == "<|im_end|>"
        return 2


def test_generate_text_passes_temperature_as_sampler(monkeypatch):
    sampler = object()
    captured = {}

    monkeypatch.setattr(
        mlx_lm.sample_utils,
        "make_sampler",
        lambda temp: captured.setdefault("temperature", temp) or sampler,
    )

    def fake_generate(model, tokenizer, **kwargs):
        assert tokenizer.eos_token_ids == {1, 2}
        captured.update(kwargs)
        return "fixed code<|im_end|>ignored garbage"

    monkeypatch.setattr(mlx_lm, "generate", fake_generate)

    assert generate_text("model", FakeTokenizer(), "Fix it") == ("fixed code", 2, 2)
    assert captured["temperature"] == 0.0
    assert captured["sampler"] is sampler
    assert "temp" not in captured

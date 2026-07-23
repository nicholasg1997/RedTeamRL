from types import SimpleNamespace

import torch

from redteamrl.sft.sft_train import sft_step


class _Tokenizer:
    eos_token_id = 3
    pad_token_id = 0

    def apply_chat_template(self, chat, add_generation_prompt, enable_thinking):
        return [1]

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [2]}


class _LossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.device = torch.device("cpu")
        self.forward_calls = 0
        self.batch_shapes = []

    def forward(self, input_ids, attention_mask, labels):
        self.forward_calls += 1
        self.batch_shapes.append(tuple(input_ids.shape))
        return SimpleNamespace(loss=self.weight.square())


_EXAMPLE = {"system": "sys", "prompt": "prompt", "target": "target"}


def _one_step(batch, microbatch_size=2):
    model = _LossModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    metrics = sft_step(model, _Tokenizer(), batch, opt, max_grad_norm=0,
                       microbatch_size=microbatch_size)
    return model, metrics


def test_duplicate_examples_do_not_multiply_effective_learning_rate():
    one_model, _ = _one_step([_EXAMPLE])
    repeated_model, _ = _one_step([_EXAMPLE] * 8)
    assert repeated_model.weight.item() == one_model.weight.item()


def test_empty_batch_is_a_noop():
    model, metrics = _one_step([])
    assert model.weight.item() == 1.0
    assert metrics == {"loss": 0.0, "n": 0}


def test_microbatching_reduces_model_calls():
    model, metrics = _one_step([_EXAMPLE] * 8, microbatch_size=2)
    assert model.forward_calls == 4
    assert model.batch_shapes == [(2, 3)] * 4
    assert metrics["n"] == 8


def test_invalid_microbatch_size_rejected():
    model = _LossModel()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    try:
        sft_step(model, _Tokenizer(), [_EXAMPLE], opt, microbatch_size=0)
    except ValueError as exc:
        assert "positive" in str(exc)
    else:
        raise AssertionError("expected ValueError")

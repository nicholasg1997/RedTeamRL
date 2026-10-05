from types import SimpleNamespace

import torch

from redteamrl.sft.example import render_target
from redteamrl.sft.sft_train import encode_example, sft_loss, sft_step


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


def test_teacher_forced_loss_does_not_update_weights_and_restores_mode():
    model = _LossModel()
    model.train()
    before = model.weight.item()
    loss = sft_loss(model, _Tokenizer(), [_EXAMPLE] * 3, microbatch_size=2)
    assert loss == 1.0
    assert model.weight.item() == before
    assert model.training is True


def test_defender_json_weights_decision_suffix_more_than_reasoning():
    example = {
        "system": "sys",
        "prompt": "prompt",
        "target": render_target({
            "reasoning": "inspect every field",
            "verdict": "redact",
            "remove": ["secret"],
        }),
    }
    _, labels, weights = encode_example(_Tokenizer(), example, decision_loss_weight=5.0)
    supervised = [weight for label, weight in zip(labels, weights) if label != -100]

    # The stub tokenizer emits one token per segment: reasoning JSON prefix, decision suffix, EOS.
    assert supervised == [1.0, 5.0, 5.0]


class _Tok:
    eos_token_id = 0

    def apply_chat_template(self, chat, add_generation_prompt, enable_thinking):
        return [1, 2]

    def __call__(self, seg, add_special_tokens=False):
        # one id per character keeps the mapping trivial to assert on
        return {"input_ids": [ord(c) % 50 + 3 for c in seg]}


def test_masked_attacker_action_gets_zero_weight_on_the_json_span():
    ex = {"system": "s", "prompt": "p",
          "target": 'too direct\n{"kind": "tool_call"}', "action_loss_weight": 0.0}
    input_ids, labels, weights = encode_example(_Tok(), ex)
    # the reasoning span 'too direct\n' has weight 1.0; the JSON span has weight 0.0; eos > 0? see below
    assert any(w == 1.0 for w in weights) and any(w == 0.0 for w in weights)
    # the trailing JSON characters must all be zero-weighted
    json_len = len('{"kind": "tool_call"}')
    assert all(w == 0.0 for w in weights[-(json_len + 1):])   # +1 for eos, also 0 when masked


def test_unmasked_attacker_action_weight_one():
    ex = {"system": "s", "prompt": "p",
          "target": 'good move\n{"kind": "submit", "value": "x"}', "action_loss_weight": 1.0}
    tok = _Tok()
    _, _, weights = encode_example(tok, ex)
    # weights[:len(prompt_ids)] are the (always-0.0) prompt positions; everything from the start of
    # the target (reasoning + JSON action + eos) must be trained at 1.0 when the action is unmasked.
    prompt_len = len(tok.apply_chat_template([], True, False))
    assert all(w == 1.0 for w in weights[prompt_len:])  # everything trained

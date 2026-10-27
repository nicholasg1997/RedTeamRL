from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from teacher_sft.encoding import encode_example, select_arm
from teacher_sft.train import DEFAULTS, batch_loss, train_loop


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 3

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, enable_thinking):
        assert tokenize is False and enable_thinking is True
        text = "".join(f"<{m['role']}>\n{m['content']}\x03" for m in messages)
        return text + ("<assistant>\n" if add_generation_prompt else "")

    def __call__(self, text, **kwargs):
        return {"input_ids": [ord(c) for c in text]}


def row(**changes):
    return dict(id="test", system="system", prompt="observation", reasoning="Think carefully.",
                final='## Plan\nCheck.\n{"kind":"message","text":"test"}') | changes


class TinyModel(torch.nn.Module):
    def __init__(self, tail=False):
        super().__init__()
        self.embedding = torch.nn.Embedding(256, 8)
        self.head = torch.nn.Linear(8, 256)
        self.config = SimpleNamespace(model_type="qwen3" if tail else "test")

    def forward(self, input_ids, attention_mask, logits_to_keep=None):
        x = self.embedding(input_ids)
        if logits_to_keep:
            x = x[:, -logits_to_keep:]
        return SimpleNamespace(logits=self.head(x))


def test_native_template_keeps_thinking_eos_and_masks_only_prompt():
    ids, labels, weights = encode_example(Tokenizer(), row())
    n = labels.index(next(v for v in labels if v != -100))
    decoded = "".join(map(chr, ids[n:]))
    assert decoded.startswith("<think>\nThink carefully.\n</think>")
    assert decoded.endswith("\x03")
    assert labels[:n] == [-100] * n and weights[:n] == [0.] * n
    assert labels[n:] == ids[n:] and weights[n:] == [1.] * (len(ids) - n)


def test_reject_truncation_and_bad_template_boundary():
    with pytest.raises(ValueError, match="refusing target truncation"):
        encode_example(Tokenizer(), row(), max_length=20)
    class BadTokenizer(Tokenizer):
        def apply_chat_template(self, messages, **kw):
            return super().apply_chat_template(messages, **kw) + ("broken" if kw["add_generation_prompt"] else "")
    with pytest.raises(ValueError, match="prefix differs"):
        encode_example(BadTokenizer(), row())


def test_arms_do_not_clone_original_failed_actions():
    rows = [row(kind="natural"), row(kind="correction")]
    assert select_arm(rows, "wins") == rows[:1]
    assert select_arm(rows, "wins+corrections") == rows
    with pytest.raises(ValueError):
        select_arm(rows, "losses")


def test_tail_logits_optimization_preserves_loss_and_gradients():
    torch.manual_seed(0)
    full, tail = TinyModel(), TinyModel(tail=True)
    tail.load_state_dict(full.state_dict())
    tok = Tokenizer()
    rows = [encode_example(tok, row()), encode_example(tok, row(prompt="longer observation"))]
    x, y = batch_loss(full, tok, rows, "cpu"), batch_loss(tail, tok, rows, "cpu")
    torch.testing.assert_close(x, y)
    x.backward()
    y.backward()
    for p, q in zip(full.parameters(), tail.parameters()):
        torch.testing.assert_close(p.grad, q.grad)


def test_microbatch_weighting_matches_full_batch_with_unequal_targets():
    torch.manual_seed(1)
    a, b = TinyModel(), TinyModel()
    b.load_state_dict(a.state_dict())
    tok = Tokenizer()
    rows = [encode_example(tok, row()), encode_example(tok, row(reasoning="A longer explanation. " * 3))]
    cfg = DEFAULTS | dict(max_steps=1, batch_size=2)
    for model, micro in ((a, 1), (b, 2)):
        train_loop(model, tok, rows, rows[:1], torch.optim.SGD(model.parameters(), lr=.01),
                   cfg | {"microbatch_size": micro}, device="cpu")
    for p, q in zip(a.parameters(), b.parameters()):
        torch.testing.assert_close(p, q)


def test_resume_matches_uninterrupted_training_and_updates_parameters():
    torch.manual_seed(2)
    a, b = TinyModel(), TinyModel()
    b.load_state_dict(a.state_dict())
    original = deepcopy(a.state_dict())
    tok = Tokenizer()
    rows = [encode_example(tok, row(reasoning=f"Reason {i}.")) for i in range(6)]
    cfg = DEFAULTS | dict(max_steps=3, batch_size=2, checkpoint_every=1)
    opt_a, opt_b = torch.optim.AdamW(a.parameters(), lr=.001), torch.optim.AdamW(b.parameters(), lr=.001)
    done = train_loop(a, tok, rows, rows[:1], opt_a, cfg, device="cpu")
    class Interrupted(Exception):
        pass
    saved = {}
    def save(model, optimizer, state):
        saved.update(model=deepcopy(model.state_dict()), optimizer=deepcopy(optimizer.state_dict()), state=deepcopy(state))
        raise Interrupted
    with pytest.raises(Interrupted):
        train_loop(b, tok, rows, rows[:1], opt_b, cfg, device="cpu", checkpoint=save)
    restored = TinyModel()
    restored.load_state_dict(saved["model"])
    opt = torch.optim.AdamW(restored.parameters(), lr=.001)
    opt.load_state_dict(saved["optimizer"])
    resumed = train_loop(restored, tok, rows, rows[:1], opt, cfg, device="cpu", state=saved["state"])
    assert resumed == done and done["step"] == 3
    assert any(not torch.equal(original[k], a.state_dict()[k]) for k in original)
    for p, q in zip(a.parameters(), restored.parameters()):
        torch.testing.assert_close(p, q)


def test_budget_stops_before_partial_target_and_requires_dev():
    model, tok = TinyModel(), Tokenizer()
    encoded = encode_example(tok, row())
    n = sum(encoded[2])
    cfg = DEFAULTS | dict(max_target_tokens=int(n) + 1, batch_size=4)
    state = train_loop(model, tok, [encoded] * 8, [encoded], torch.optim.SGD(model.parameters(), lr=.01), cfg, device="cpu")
    assert state["cursor"] == state["step"] == 1 and state["target_tokens_seen"] == n
    with pytest.raises(ValueError, match="Both train and dev"):
        train_loop(model, tok, [encoded], [], None, cfg, device="cpu")


def test_nonfinite_loss_never_advances_checkpoint():
    model, tok = TinyModel(), Tokenizer()
    with torch.no_grad():
        model.head.weight.fill_(float("nan"))
    examples = [encode_example(tok, row())]
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        train_loop(model, tok, examples, examples, torch.optim.SGD(model.parameters(), lr=.01),
                   DEFAULTS, device="cpu", checkpoint=lambda *a: pytest.fail("must not save"))


def test_actual_qwen_template_and_tiny_qwen_update_when_tokenizer_available():
    import os
    path = os.environ.get("TEACHER_SFT_TEST_TOKENIZER")
    if not path:
        pytest.skip("Set TEACHER_SFT_TEST_TOKENIZER to a local Qwen3 tokenizer; never downloads in tests")
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(path, local_files_only=True)
    example = encode_example(tok, row())
    target = tok.decode([v for v in example[1] if v != -100])
    assert target.count("<think>") == target.count("</think>") == 1
    assert "<|im_end|>" in target
    model = Qwen3ForCausalLM(Qwen3Config(vocab_size=len(tok), hidden_size=16, intermediate_size=32,
        num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=1, head_dim=8))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
    state = train_loop(model, tok, [example], [example], optimizer,
                       DEFAULTS | {"max_steps": 1}, device="cpu")
    assert state["history"][0]["gradient_norm"] > 0
    assert state["dev_loss_after"] < state["dev_loss_before"]

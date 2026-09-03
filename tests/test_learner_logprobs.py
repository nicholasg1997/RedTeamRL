"""logprobs must not materialize logits for positions it throws away.

The attacker phase OOM'd on an 80GB A100 inside learner.logprobs. Qwen3's vocab is ~152k, so a
15k-token sequence (a 40k-char transcript plus a 2048-token completion) produces a 4.25 GiB logits
tensor -- of which only the ~2048 completion positions are ever used. The defender phase never hit
it because its prompts are ~2.5k tokens.
"""
import contextlib
from types import SimpleNamespace

import torch

from redteamrl.train.learner import Learner

VOCAB = 11


class _FakeLM:
    """Deterministic, position-dependent logits, so an off-by-one in the slice cannot pass."""
    device = "cpu"

    def __init__(self):
        self.calls = []

    def _full(self, length):
        return (torch.arange(length * VOCAB, dtype=torch.float32).reshape(1, length, VOCAB) % 7)

    def __call__(self, ids, attention_mask=None, logits_to_keep=None):
        length = ids.shape[1]
        self.calls.append(logits_to_keep)
        logits = self._full(length)
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:, :]
        return SimpleNamespace(logits=logits)

    def disable_adapter(self):
        return contextlib.nullcontext()


def _reference(model, prompt_ids, completion_ids):
    """What the naive full-sequence implementation computed."""
    full = model._full(len(prompt_ids) + len(completion_ids))[0]
    n_prompt, n_comp = len(prompt_ids), len(completion_ids)
    sliced = full[n_prompt - 1: n_prompt - 1 + n_comp]
    logp = torch.log_softmax(sliced, dim=-1)
    return logp.gather(1, torch.tensor(completion_ids).unsqueeze(1)).squeeze(1)


def test_logprobs_match_the_full_sequence_computation():
    model = _FakeLM()
    learner = Learner(model, tokenizer=None, optimizer=None)
    prompt, completion = [1, 2, 3, 4, 5], [6, 7, 8]
    got = learner.logprobs(prompt, completion, use_adapter=True, with_grad=False)
    torch.testing.assert_close(got, _reference(model, prompt, completion))


def test_only_the_completion_positions_are_materialized():
    model = _FakeLM()
    learner = Learner(model, tokenizer=None, optimizer=None)
    prompt, completion = list(range(20)), [1, 2, 3]
    learner.logprobs(prompt, completion, use_adapter=True, with_grad=False)
    # one extra position: the completion's first token is predicted FROM the prompt's last one
    assert model.calls == [len(completion) + 1]


def test_single_token_completion_is_not_off_by_one():
    model = _FakeLM()
    learner = Learner(model, tokenizer=None, optimizer=None)
    prompt, completion = [1, 2, 3], [9]
    got = learner.logprobs(prompt, completion, use_adapter=True, with_grad=False)
    torch.testing.assert_close(got, _reference(model, prompt, completion))

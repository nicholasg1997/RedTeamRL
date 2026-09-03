"""Long-context backward passes need gradient checkpointing, and PEFT makes it a 3-part setup.

Qwen3-8B saves ~1.14 GiB of activations per layer at 15k tokens; across 36 layers that is 41.2 GiB
for ONE backward pass, on top of 16.4 GiB of weights. No memory-fraction tuning fits that on an
80GB card shared with two vLLM engines. Checkpointing trades ~30% compute for ~10x the memory.
"""
from types import SimpleNamespace

from redteamrl.train.learner import prepare_for_long_context_training


class _FakePeftModel:
    def __init__(self):
        self.config = SimpleNamespace(use_cache=True)
        self.calls = []

    def gradient_checkpointing_enable(self, **kwargs):
        self.calls.append("gradient_checkpointing_enable")

    def enable_input_require_grads(self):
        self.calls.append("enable_input_require_grads")


def test_enables_gradient_checkpointing():
    model = _FakePeftModel()
    prepare_for_long_context_training(model)
    assert "gradient_checkpointing_enable" in model.calls


def test_enables_input_require_grads():
    """THE PEFT gotcha. The base model is frozen, so a checkpointed segment's inputs do not
    require grad, the recomputed graph is detached, and backward silently produces no LoRA
    gradient at all -- a training run that looks healthy and learns nothing."""
    model = _FakePeftModel()
    prepare_for_long_context_training(model)
    assert "enable_input_require_grads" in model.calls


def test_disables_kv_cache():
    """use_cache is incompatible with checkpointing; transformers warns and silently disables it."""
    model = _FakePeftModel()
    prepare_for_long_context_training(model)
    assert model.config.use_cache is False


def test_returns_the_model_for_chaining():
    model = _FakePeftModel()
    assert prepare_for_long_context_training(model) is model


def test_tolerates_a_model_without_enable_input_require_grads():
    """Not every model exposes it; the helper must not crash on one that does not."""
    class _Minimal:
        def __init__(self):
            self.config = SimpleNamespace(use_cache=True)
            self.enabled = False
        def gradient_checkpointing_enable(self, **kwargs):
            self.enabled = True
    model = _Minimal()
    prepare_for_long_context_training(model)
    assert model.enabled and model.config.use_cache is False

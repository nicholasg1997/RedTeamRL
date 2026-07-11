"""MLX-backed `generate(system, messages) -> str` adapter (Apple Silicon, local inference).

Kept out of core deps — mlx-lm is Apple-only (see pyproject [local] + [tool.uv] override).
Import this lazily (inside a function / __main__), not at module top level, so the rest of
the package still imports on a non-Apple / CUDA box that has no mlx-lm installed.

Qwen3 model-card sampling recommendations:
  - thinking     (enable_thinking=True):  temp=0.6, top_p=0.95, top_k=20, min_p=0
  - non-thinking (enable_thinking=False): temp=0.7, top_p=0.8,  top_k=20, min_p=0
"""
from mlx_lm import load, generate as _mlx_generate
from mlx_lm.sample_utils import make_sampler


def make_mlx_generate(
    model_id: str,
    max_tokens: int = 512,
    temp: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    enable_thinking: bool = False,
):
    """Load a model once and return a `generate(system, messages) -> str` closure.
    """
    model, tokenizer = load(model_id)
    sampler = make_sampler(temp=temp, top_p=top_p, top_k=top_k, min_p=min_p)

    def generate(system: str, messages: list[dict]) -> str:
        chat = [{"role": "system", "content": system}, *messages]
        prompt = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        return _mlx_generate(
            model, tokenizer, prompt=prompt, max_tokens=max_tokens, sampler=sampler, verbose=False
        )

    return generate

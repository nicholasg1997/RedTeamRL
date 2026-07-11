"""MLX-backed `generate(system, messages) -> str` adapter (Apple Silicon, local inference).

Kept out of core deps — mlx-lm is Apple-only (see pyproject [local] + [tool.uv] override).
Import this lazily (inside a function / __main__), not at module top level, so the rest of
the package still imports on a non-Apple / CUDA box that has no mlx-lm installed.
"""
from mlx_lm import load, generate as _mlx_generate


def make_mlx_generate(model_id: str, max_tokens: int = 512, enable_thinking: bool = False):
    """Load a model once and return a `generate(system, messages) -> str` closure."""
    model, tokenizer = load(model_id)

    def generate(system: str, messages: list[dict]) -> str:
        chat = [{"role": "system", "content": system}, *messages]
        prompt = tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, enable_thinking=enable_thinking,
            temperature=0.6, top_p=0.95, top_k=20
        )
        return _mlx_generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)

    return generate


"""
For thinking mode (enable_thinking=True), use Temperature=0.6, TopP=0.95, TopK=20, and MinP=0. DO NOT use greedy decoding, as it can lead to performance degradation and endless repetitions.
For non-thinking mode (enable_thinking=False), we suggest using Temperature=0.7, TopP=0.8, TopK=20, and MinP=0.
"""

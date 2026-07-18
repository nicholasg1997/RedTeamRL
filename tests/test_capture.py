from redteamrl.train.capture import CapturingGenerate, Example


class _FakeTokenizer:
    def apply_chat_template(self, msgs, add_generation_prompt, return_tensors=None):
        # deterministic: 3 prompt tokens
        import torch
        return torch.tensor([[1, 2, 3]])

    def decode(self, ids, skip_special_tokens=True):
        return "DECODED:" + ",".join(str(int(i)) for i in ids)


class _FakeModel:
    device = "cpu"
    def generate(self, input_ids, **kw):
        import torch
        # echo prompt + two new tokens [7, 8]
        return torch.tensor([[1, 2, 3, 7, 8]])


def test_capture_appends_example_and_returns_text():
    gen = CapturingGenerate(_FakeModel(), _FakeTokenizer(), temperature=0.7)
    out = gen("SYS", [{"role": "user", "content": "hi"}])
    assert len(gen.buffer) == 1
    ex = gen.buffer[0]
    assert ex.prompt_ids == [1, 2, 3]
    assert ex.completion_ids == [7, 8]          # only the NEW tokens
    assert out == "DECODED:7,8"                  # returns decoded completion only


def test_each_call_appends_one():
    gen = CapturingGenerate(_FakeModel(), _FakeTokenizer())
    gen("s", [{"role": "user", "content": "a"}])
    gen("s", [{"role": "user", "content": "b"}])
    assert len(gen.buffer) == 2
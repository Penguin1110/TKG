from tkg.experiment.open_weight_abcd_engineering_v25 import (
    ARM_LOCAL_EXPANSIONS, ARM_WIDTHS, MicrobatchedLogprobBackendV25,
)


class _FakeBackend:
    backend_name = "fake"

    def conditional_token_logprobs(self, prompt: str, continuation: str):
        return [float(len(prompt)), float(len(continuation))]

    def conditional_token_logprobs_batch(self, prompt: str, continuations: list[str]):
        return [self.conditional_token_logprobs(prompt, value) for value in continuations]

    def generate_text(self, prompt: str, *, max_new_tokens=192, system_prompt=None):
        return f"{system_prompt}:{prompt}:{max_new_tokens}"


def test_microbatch_preserves_order_and_exact_values():
    original = _FakeBackend()
    wrapped = MicrobatchedLogprobBackendV25(original, 2)
    values = ["a", "bb", "ccc", "dddd", "eeeee"]
    assert wrapped.conditional_token_logprobs_batch("prompt", values) == (
        original.conditional_token_logprobs_batch("prompt", values)
    )
    assert wrapped.generate_text("x", max_new_tokens=3, system_prompt="s") == "s:x:3"


def test_development_arm_local_expansion_contract():
    assert ARM_WIDTHS == {"A": None, "B": 1, "C": 3, "D": 5}
    assert ARM_LOCAL_EXPANSIONS == {"A": None, "B": 1, "C": 3, "D": 5}

from __future__ import annotations

from tkg.experiment.pipeline_policy_v24 import authorize_pipeline_v24


def test_failed_controller_does_not_lock_data_line() -> None:
    policy = authorize_pipeline_v24(
        api_controller_gate_passed=False,
        open_weight_method_gate_passed=False,
        admitted_case_count=0,
    )
    assert policy.fresh_case_generation is True
    assert policy.machine_validation is True
    assert policy.pk_admission is True
    assert policy.api_abcd is False
    assert policy.open_weight_abcd is False


def test_each_abcd_arm_requires_its_method_gate_and_admitted_cases() -> None:
    policy = authorize_pipeline_v24(
        api_controller_gate_passed=False,
        open_weight_method_gate_passed=True,
        admitted_case_count=3,
    )
    assert policy.api_abcd is False
    assert policy.open_weight_abcd is True

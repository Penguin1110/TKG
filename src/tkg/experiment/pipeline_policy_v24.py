"""Independent data-line and method-line gates for temporal evaluation v2.4."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


PIPELINE_POLICY_SCHEMA_V24 = "temporal-evaluation-pipeline-policy-v2.4"


@dataclass(frozen=True)
class PipelineAuthorizationV24:
    fresh_case_generation: bool
    machine_validation: bool
    pk_admission: bool
    api_abcd: bool
    open_weight_abcd: bool
    reasons: tuple[str, ...]
    schema_version: str = PIPELINE_POLICY_SCHEMA_V24

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def authorize_pipeline_v24(
    *, api_controller_gate_passed: bool,
    open_weight_method_gate_passed: bool,
    admitted_case_count: int,
) -> PipelineAuthorizationV24:
    """Keep data admission independent; require both lines only for A/B/C/D."""
    if admitted_case_count < 0:
        raise ValueError("admitted_case_count must be non-negative")
    has_cases = admitted_case_count > 0
    reasons = [
        "data_line_is_independent_of_solver_controller_gate",
    ]
    if not has_cases:
        reasons.append("abcd_locked_no_pk_admitted_cases")
    if not api_controller_gate_passed:
        reasons.append("api_abcd_locked_controller_gate_failed")
    if not open_weight_method_gate_passed:
        reasons.append("open_weight_abcd_locked_method_gate_not_passed")
    return PipelineAuthorizationV24(
        fresh_case_generation=True,
        machine_validation=True,
        pk_admission=True,
        api_abcd=api_controller_gate_passed and has_cases,
        open_weight_abcd=open_weight_method_gate_passed and has_cases,
        reasons=tuple(reasons),
    )

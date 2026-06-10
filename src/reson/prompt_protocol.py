from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class PromptPhase:
    name: str
    duration_s: float
    label: str | None = None


@dataclass(frozen=True)
class PhasePosition:
    index: int
    phase: PromptPhase
    elapsed_in_phase_s: float
    remaining_in_phase_s: float


def build_protocol(
    *,
    settle_sec: float,
    rest_sec: float,
    trials: int,
    press_sec: float,
    gap_sec: float,
    final_rest_sec: float,
    artifact_sec: float,
) -> list[PromptPhase]:
    phases: list[PromptPhase] = []
    if settle_sec > 0:
        phases.append(PromptPhase("SETTLE", settle_sec))
    if rest_sec > 0:
        phases.append(PromptPhase("REST", rest_sec))
    for trial in range(max(trials, 0)):
        phases.append(PromptPhase(f"CLICK {trial + 1}/{trials}", press_sec, label="CLICK"))
        if trial != trials - 1 and gap_sec > 0:
            phases.append(PromptPhase("REST", gap_sec))
    if artifact_sec > 0:
        phases.append(PromptPhase("ARTIFACT_NO_LABEL", artifact_sec))
    if final_rest_sec > 0:
        phases.append(PromptPhase("REST", final_rest_sec))
    return phases


def protocol_duration_s(phases: list[PromptPhase]) -> float:
    return sum(phase.duration_s for phase in phases)


def phase_at(phases: list[PromptPhase], elapsed_s: float) -> PhasePosition | None:
    cursor = 0.0
    for idx, phase in enumerate(phases):
        next_cursor = cursor + phase.duration_s
        if elapsed_s < next_cursor:
            return PhasePosition(
                index=idx,
                phase=phase,
                elapsed_in_phase_s=max(0.0, elapsed_s - cursor),
                remaining_in_phase_s=max(0.0, next_cursor - elapsed_s),
            )
        cursor = next_cursor
    return None


def protocol_as_dicts(phases: list[PromptPhase]) -> list[dict[str, object]]:
    return [asdict(phase) for phase in phases]

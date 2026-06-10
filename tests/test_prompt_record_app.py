from reson.apps.prompt_record_app import build_parser
from reson.prompt_protocol import build_protocol, phase_at, protocol_duration_s


def test_prompt_protocol_creates_interval_labels_without_key_binding():
    phases = build_protocol(
        settle_sec=1.0,
        rest_sec=2.0,
        trials=3,
        press_sec=0.5,
        gap_sec=1.0,
        final_rest_sec=2.0,
        artifact_sec=4.0,
    )

    click_phases = [phase for phase in phases if phase.label == "CLICK"]

    assert len(click_phases) == 3
    assert phases[0].name == "SETTLE"
    assert phases[-2].name == "ARTIFACT_NO_LABEL"
    assert phases[-1].name == "REST"
    assert protocol_duration_s(phases) == 12.5
    assert phase_at(phases, 3.1).phase.name == "CLICK 1/3"


def test_prompt_record_parser_defaults_to_no_keyboard_labeling():
    args = build_parser().parse_args([])

    assert args.trials == 20
    assert args.press_sec == 1.0
    assert args.gap_sec == 3.0
    assert args.artifact_sec == 0.0

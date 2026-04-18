from reson.pipeline import run_switch_pipeline
from reson.types import EdgeEvent


class _FakeReader:
    def iter_lines(self):
        yield "0 1000 0\n"
        yield "10 1010 0\n"


class _FakeDetector:
    def __init__(self):
        self._events: list[EdgeEvent] = []

    def update(self, sample):
        if sample.t_ms == 0:
            self._events.append(
                EdgeEvent(
                    state="light",
                    start_ms=sample.t_ms,
                    end_ms=sample.t_ms,
                    duration_ms=0,
                    phase="down",
                    press_class="light",
                )
            )
        else:
            self._events.append(
                EdgeEvent(
                    state="light",
                    start_ms=0,
                    end_ms=sample.t_ms,
                    duration_ms=sample.t_ms,
                    phase="up",
                    press_class="light",
                )
            )
        return "light"

    def pop_events(self):
        out = self._events
        self._events = []
        return out

    def flush(self, final_t_ms):
        return []


def test_switch_pipeline_emits_binary_down_up_events():
    events = list(run_switch_pipeline(_FakeReader(), _FakeDetector()))

    assert [event.phase for event in events] == ["down", "up"]
    assert events[0].t_ms == 0
    assert events[1].duration_ms == 10

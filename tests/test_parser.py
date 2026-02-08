from reson.parser import parse_line


def test_parse_valid_line():
    sample = parse_line("123 456 78")
    assert sample is not None
    assert sample.t_ms == 123
    assert sample.raw == 456
    assert sample.env == 78


def test_parse_with_extra_spaces():
    sample = parse_line("  1   2   3   ")
    assert sample is not None
    assert (sample.t_ms, sample.raw, sample.env) == (1, 2, 3)


def test_parse_invalid_tokens_returns_none():
    assert parse_line("a b c") is None
    assert parse_line("1 2") is None
    assert parse_line("") is None

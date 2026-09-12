import pytest

from diagram_restore.latency import measure_latency


def test_measure_latency_reports_the_requested_repeat_count():
    calls = []
    result = measure_latency(lambda: calls.append(1), warmup=3, repeats=5)
    assert result["repeats"] == 5
    assert result["warmup"] == 3
    assert len(calls) == 8


def test_measure_latency_reports_nonnegative_timing_statistics():
    result = measure_latency(lambda: None, warmup=1, repeats=4)
    assert result["median_seconds"] >= 0
    assert result["p95_seconds"] >= result["median_seconds"] >= 0
    assert result["mean_seconds"] >= 0


@pytest.mark.parametrize("warmup,repeats", [(-1, 5), (0, 0)])
def test_measure_latency_rejects_invalid_counts(warmup, repeats):
    with pytest.raises(ValueError):
        measure_latency(lambda: None, warmup=warmup, repeats=repeats)

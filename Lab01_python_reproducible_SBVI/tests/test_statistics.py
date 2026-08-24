import numpy as np
import pytest

from python_scientific.descriptive_statistics import descriptive_statistics


def test_statistics_known_values():
    result = descriptive_statistics([1.0, 2.0, 3.0, 4.0])

    assert result["count (muestras)"] == 4
    assert result["mean"] == pytest.approx(2.5)
    assert result["median"] == pytest.approx(2.5)
    assert result["sample_standard_deviation"] == pytest.approx(1.2909944487)
    assert result["minimum"] == pytest.approx(1.0)
    assert result["maximum"] == pytest.approx(4.0)
    assert result["q1"] == pytest.approx(1.75)
    assert result["q3"] == pytest.approx(3.25)
    assert result["iqr"] == pytest.approx(1.5)
    assert result["rms"] == pytest.approx(2.7386127875)


def test_statistics_single_value():
    result = descriptive_statistics([5.0])

    assert result["count (muestras)"] == 1
    assert result["mean"] == pytest.approx(5.0)
    assert result["median"] == pytest.approx(5.0)
    assert result["minimum"] == pytest.approx(5.0)
    assert result["maximum"] == pytest.approx(5.0)


def test_empty_signal():
    with pytest.raises(ValueError):
        descriptive_statistics([])


def test_statistics_non_finite_values():
    with pytest.raises(ValueError):
        descriptive_statistics([1.0, np.nan, 3.0])

    with pytest.raises(ValueError):
        descriptive_statistics([1.0, np.inf, 3.0])


def test_statistics_invalid_shape():
    with pytest.raises(ValueError):
        descriptive_statistics([[1.0, 2.0], [3.0, 4.0]])

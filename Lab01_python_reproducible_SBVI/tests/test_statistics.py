import numpy as np
import pytest

from python_scientific.descriptive_statistics import descriptive_statistics


def test_statistics_known_values():
    result = descriptive_statistics([1.0, 2.0, 3.0, 4.0], unit="mV")

    assert result["count (muestras)"] == 4
    assert result["mean (mV)"] == pytest.approx(2.5)
    assert result["median (mV)"] == pytest.approx(2.5)
    assert result["sample_standard_deviation (mV)"] == pytest.approx(1.2909944487)
    assert result["minimum (mV)"] == pytest.approx(1.0)
    assert result["maximum (mV)"] == pytest.approx(4.0)
    assert result["q1 (mV)"] == pytest.approx(1.75)
    assert result["q3 (mV)"] == pytest.approx(3.25)
    assert result["iqr (mV)"] == pytest.approx(1.5)
    assert result["rms (mV)"] == pytest.approx(2.7386127875)


def test_statistics_single_value():
    result = descriptive_statistics([5.0], unit="mV")

    assert result["count (muestras)"] == 1
    assert result["mean (mV)"] == pytest.approx(5.0)
    assert result["median (mV)"] == pytest.approx(5.0)
    assert result["minimum (mV)"] == pytest.approx(5.0)
    assert result["maximum (mV)"] == pytest.approx(5.0)


def test_empty_signal():
    with pytest.raises(ValueError):
        descriptive_statistics([], unit="mV")


def test_statistics_non_finite_values():
    with pytest.raises(ValueError):
        descriptive_statistics([1.0, np.nan, 3.0], unit="mV")

    with pytest.raises(ValueError):
        descriptive_statistics([1.0, np.inf, 3.0], unit="mV")


def test_statistics_invalid_shape():
    with pytest.raises(ValueError):
        descriptive_statistics([[1.0, 2.0], [3.0, 4.0]], unit="mV")

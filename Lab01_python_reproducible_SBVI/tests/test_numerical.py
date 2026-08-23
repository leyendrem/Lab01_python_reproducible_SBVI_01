import numpy as np
import pytest

from python_scientific.numerical import (
    cumulative_trapezoidal_integral,
    numerical_derivative,
    trapezoidal_integral,
)


def test_derivative():
    time = np.linspace(0, 1, 1001)
    signal = np.sin(2 * np.pi * time)

    derivative = numerical_derivative(signal, time)
    expected = 2 * np.pi * np.cos(2 * np.pi * time)

    interior_error = np.abs(derivative[1:-1] - expected[1:-1])

    assert np.max(interior_error) < 0.01


def test_integral():
    time = np.linspace(0, 1, 1001)
    signal = time**2

    result = trapezoidal_integral(signal, time)

    assert abs(result - 1 / 3) < 0.00001


def test_derivative_invalid_lengths():
    values = np.array([1.0, 2.0, 3.0])
    time = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        numerical_derivative(values, time)


def test_derivative_too_few_samples():
    values = np.array([1.0, 2.0])
    time = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        numerical_derivative(values, time)


def test_integral_invalid_lengths():
    values = np.array([1.0, 2.0, 3.0])
    time = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        trapezoidal_integral(values, time)


def test_cumulative_integral():
    time = np.linspace(0, 1, 1001)
    signal = time**2

    result = cumulative_trapezoidal_integral(signal, time)

    assert result.shape == signal.shape
    assert result[0] == pytest.approx(0.0)
    assert result[-1] == pytest.approx(
        1 / 3,
        abs=1e-5,
    )


def test_cumulative_integral_invalid_lengths():
    values = np.array([1.0, 2.0, 3.0])
    time = np.array([0.0, 1.0])

    with pytest.raises(ValueError):
        cumulative_trapezoidal_integral(values, time)

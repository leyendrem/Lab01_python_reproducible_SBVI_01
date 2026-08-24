import numpy as np
import pytest

from python_scientific.simulation import (
    center_signal,
    generate_synthetic_signal,
    get_algebra_data,
)

n_samples = 500
duration = 10.0
frequency = 0.5
seed = 42


def test_simulation_shape():
    time_s, signal_mv = generate_synthetic_signal(
        n_samples=n_samples,
        duration=duration,
        frequency=frequency,
        seed=seed,
    )

    assert time_s.shape == signal_mv.shape
    assert signal_mv.ndim == 1
    assert signal_mv.size == n_samples


def test_simulation_time_axis():
    time_s, _ = generate_synthetic_signal(
        n_samples=n_samples,
        duration=duration,
        frequency=frequency,
        seed=seed,
    )

    assert time_s[0] == 0.0
    assert time_s[-1] == duration
    assert np.all(np.diff(time_s) > 0)

    sampling_frequency_hz = 1 / np.mean(np.diff(time_s))
    assert sampling_frequency_hz == pytest.approx(50.0, abs=0.2)


def test_simulation_same_seed():
    time_1, signal_1 = generate_synthetic_signal(
        n_samples=n_samples,
        duration=duration,
        frequency=frequency,
        seed=seed,
    )

    time_2, signal_2 = generate_synthetic_signal(
        n_samples=n_samples,
        duration=duration,
        frequency=frequency,
        seed=seed,
    )

    assert np.array_equal(time_1, time_2)
    assert np.array_equal(signal_1, signal_2)


def test_simulation_different_seed():
    _, signal_1 = generate_synthetic_signal(
        n_samples=n_samples, duration=duration, frequency=frequency, seed=42
    )
    _, signal_2 = generate_synthetic_signal(
        n_samples=n_samples, duration=duration, frequency=frequency, seed=123
    )

    assert not np.array_equal(signal_1, signal_2)


def test_center_signal_has_zero_mean():
    _, signal_mv = generate_synthetic_signal(
        n_samples=n_samples,
        duration=duration,
        frequency=frequency,
        seed=seed,
    )

    centered_signal = center_signal(signal_mv)

    assert np.mean(centered_signal) == pytest.approx(0.0, abs=1e-7)


def test_center_signal_empty_error():
    with pytest.raises(ValueError, match="La señal no puede estar vacía"):
        center_signal(np.array([]))


def test_get_algebra_data():
    time_s = np.linspace(0, 10, 500)
    A_matrix, x_channels = get_algebra_data(time_s)

    assert A_matrix.shape == (2, 2)
    assert x_channels.shape == (2, 100)

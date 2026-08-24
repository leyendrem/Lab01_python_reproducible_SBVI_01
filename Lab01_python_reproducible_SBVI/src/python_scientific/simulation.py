from collections.abc import Callable

import numpy as np


def generate_synthetic_signal(
    n_samples: int,
    duration: float,
    frequency: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Genera una señal sintética reproducible."""

    if n_samples < 2:
        raise ValueError("Se requieren al menos dos muestras.")
    if duration <= 0:
        raise ValueError("La duración debe ser positiva.")
    if frequency < 0:
        raise ValueError("La frecuencia debe ser mayor a cero.")

    time = np.linspace(0, duration, n_samples)

    rng = np.random.default_rng(seed)

    signal_mv = (
        0.15
        + np.sin(2 * np.pi * frequency * time)
        + 0.25 * np.sin(2 * np.pi * 10 * time)
        + 0.08 * rng.normal(size=n_samples)
    )

    return time, signal_mv


def center_signal(signal_mv: np.ndarray) -> np.ndarray:
    """Resta la media de la señal."""

    signal = np.asarray(signal_mv, dtype=float)

    if signal.size == 0:
        raise ValueError("La señal no puede estar vacía.")

    return signal - np.mean(signal)


def run_channel_validation(validate: Callable) -> dict[str, object]:
    """Ejecuta los casos de prueba para la validación de canales."""

    observed_1 = ["SYNTH_A", "SYNTH_B", "SYNTH_A"]
    required_1 = {"SYNTH_A", "SYNTH_B", "REFERENCE"}
    ans_1 = validate(observed_1, required_1)

    try:
        observed_2 = ["SYNTH_A", " "]
        required_2 = {"SYNTH_A", "SYNTH_B"}
        validate(observed_2, required_2)
        ans_2 = "No se lanzó error esperado"
    except ValueError as error:
        ans_2 = f"Error capturado exitosamente: {error!s}"

    return {"caso_principal": ans_1, "caso_limite_vacio": ans_2}


def get_algebra_data(time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Genera las matrices y canales de prueba para álgebra lineal."""

    A_matrix = np.array([[0.10, 1.0], [1.0, 0.20]])
    c1 = np.sin(2 * np.pi * 0.05 * time[:100])
    c2 = np.cos(2 * np.pi * 0.1 * time[:100])
    x_channels = np.array([c1, c2])

    return A_matrix, x_channels

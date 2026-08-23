import matplotlib.pyplot as plt
import numpy as np
import pytest

from python_scientific.three_panel_figure import three_panel_figure


def test_three_panel_figure_success():
    # Creamos datos de tiempo y señales válidas de prueba.
    time = np.linspace(0, 10, 1000)
    raw_signal = np.sin(2 * np.pi * time) + 1.5
    derivative = 2 * np.pi * np.cos(2 * np.pi * time)

    fig = three_panel_figure(
        time=time,
        raw_signal=raw_signal,
        derivative=derivative,
        derivative_start=1.0,
        derivative_end=2.0,
    )

    # Verificamos que devuelva una figura de Matplotlib y contenga los 3 subplots esperados.
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3

    plt.close(fig)


def test_three_panel_figure_invalid_dimensions():
    # Arreglos de distintas longitudes para probar que lance ValueError.
    time = np.linspace(0, 3, 300)
    raw_signal = np.sin(2 * np.pi * time)
    derivative = np.array([1.0, 2.0, 3.0])  # Longitud incorrecta.

    with pytest.raises(ValueError):
        three_panel_figure(time, raw_signal, derivative)


def test_three_panel_figure_empty_array():
    # Arreglos vacíos.
    empty_arr = np.array([])

    with pytest.raises(ValueError):
        three_panel_figure(empty_arr, empty_arr, empty_arr)


def test_three_panel_figure_invalid_interval():
    # Intervalo de recorte incorrecto.
    time = np.linspace(0, 10, 1000)
    raw_signal = np.sin(2 * np.pi * time)
    derivative = np.cos(2 * np.pi * time)

    with pytest.raises(ValueError):
        three_panel_figure(
            time=time,
            raw_signal=raw_signal,
            derivative=derivative,
            derivative_start=2.0,
            derivative_end=1.0,  # Inicio > Final.
        )

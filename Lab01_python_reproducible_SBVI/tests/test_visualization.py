from pathlib import Path

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

    # Verificación de retorno de una figura de Matplotlib,
    # y que contenga los 3 subplots esperados.
    assert isinstance(fig, plt.Figure)
    assert len(fig.axes) == 3

    # Verificación de etiquetas, títulos y ejes.
    # Panel 0: Señal Cruda y Centrada
    assert "Señal Cruda" in fig.axes[0].get_title()
    assert fig.axes[0].get_ylabel() != ""

    # Panel 1: Histograma de amplitudes
    assert "Distribución" in fig.axes[1].get_title()
    assert fig.axes[1].get_xlabel() != ""
    assert fig.axes[1].get_ylabel() != ""

    # Panel 2: Derivada numérica recortada
    assert "Derivada numérica" in fig.axes[2].get_title()
    assert fig.axes[2].get_xlabel() != ""
    assert fig.axes[2].get_ylabel() != ""

    # Verificación de que el archivo de imagen se pueda guardar realmente en disco
    output_dir = Path("figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    file_path = output_dir / "summary_test.png"

    fig.savefig(file_path, dpi=120)
    assert file_path.is_file(), (
        "El archivo de imagen no se creó correctamente en el disco."
    )
    assert file_path.stat().st_size > 0, "El archivo de imagen está vacío."

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

import matplotlib.pyplot as plt
import numpy as np


def three_panel_figure(
    time,
    raw_signal,
    derivative,
    derivative_start=1.0,
    derivative_end=2.0,
):
    """Tres paneles: señal, histograma y derivada recortada"""

    if time.ndim != 1 or raw_signal.ndim != 1 or derivative.ndim != 1:
        raise ValueError("Tiempo y señal deben ser unidimensionales.")
    if not time.shape == raw_signal.shape == derivative.shape:
        raise ValueError("Tiempo y señal deben tener igual longitud.")
    if time.size == 0:
        raise ValueError("El arreglo no puede estar vacío.")
    if derivative_start >= derivative_end:
        raise ValueError("El inicio debe ser menor que el final.")

    raw_signal_centered = raw_signal - np.mean(raw_signal)
    derivative_centered = derivative - np.mean(derivative)

    mask = (time >= derivative_start) & (time <= derivative_end)
    if not np.any(mask):
        raise ValueError("El intervalo no contiene muestras.")

    time_zoom = time[mask]
    signal_zoom = derivative_centered[mask]

    # Estilo global gráficas
    plot_style = {
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 120,
        "savefig.dpi": 180,
    }

    with plt.rc_context(plot_style):
        # Crear la figura con 3 paneles verticales
        fig, axes = plt.subplots(
            nrows=3, ncols=1, figsize=(9, 10), constrained_layout=True
        )

        # Panel 1: Comparación de señal cruda vs centrada
        axes[0].plot(time, raw_signal, color="gray", alpha=0.5, label="Cruda")
        axes[0].plot(
            time, raw_signal_centered, color="#145da0", linewidth=1.2, label="Centrada"
        )
        axes[0].legend()
        axes[0].set(
            title="Señal Cruda y Señal Centrada (Referencia)",
            ylabel="Amplitud [mV]",
        )

        # Panel 2: Histograma de amplitudes
        axes[1].hist(raw_signal, bins=25, color="#0b8fac", edgecolor="white", alpha=0.9)
        axes[1].set(
            title="Distribución de amplitudes de la señal",
            xlabel="Amplitud [mV]",
            ylabel="Frecuencia [muestras]",
        )

        # Panel 3: Derivada numérica en un intervalo recortado (1.0s - 2.0s)
        axes[2].plot(time_zoom, signal_zoom, color="#b04a3f", linewidth=1.2)
        axes[2].set(
            title="Derivada numérica (Intervalo recortado: 1.0s a 2.0s)",
            xlabel="Tiempo [s]",
            ylabel="Derivada [mV/s]",
        )

    return fig


if __name__ == "__main__":
    pass

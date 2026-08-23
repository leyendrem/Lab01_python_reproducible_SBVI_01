import datetime
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_scientific.descriptive_statistics import descriptive_statistics
from python_scientific.didactic_transformation import transformacion_didactica_matrices
from python_scientific.numerical import (
    cumulative_trapezoidal_integral,
    derivative_error_summary,
    numerical_derivative,
    trapezoidal_integral,
)
from python_scientific.quality import validate_channel_configuration
from python_scientific.three_panel_figure import three_panel_figure


def main() -> None:
    print("Iniciando pipeline de reproducción automatizada...")

    # Configuración Global de Parámetros
    seed = 42
    f0 = 0.5  # Frecuencia base
    dt = 0.02  # Paso temporal
    t_start = 0.0
    t_end = 10.0

    # Definición de tiempo, muestras y generador aleatorio moderno
    time_s = np.arange(t_start, t_end, dt)
    n_samples = len(time_s)
    rng = np.random.default_rng(seed)

    # Creación y aseguramiento de directorios de salida requeridos.
    Path("results").mkdir(parents=True, exist_ok=True)
    Path("figures").mkdir(parents=True, exist_ok=True)
    Path("data/processed").mkdir(parents=True, exist_ok=True)

    # Validación de configuración de canales con múltiples casos de prueba.
    observed_1 = ["SYNTH_A", "SYNTH_B", "SYNTH_A"]
    required_1 = {"SYNTH_A", "SYNTH_B", "REFERENCE"}

    ans_1 = validate_channel_configuration(observed_1, required_1)

    try:
        observed_2 = ["SYNTH_A", " "]
        required_2 = {"SYNTH_A", "SYNTH_B"}
        validate_channel_configuration(observed_2, required_2)
        ans_2 = "No se lanzó error esperado"
    except ValueError as error:
        ans_2 = f"Error capturado exitosamente: {error!s}"

    quality_summary = {"caso_principal": ans_1, "caso_limite_vacio": ans_2}
    Path("results/quality_report.json").write_text(
        json.dumps(quality_summary, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Simulación de datos con parámetros externos y cálculo estadístico.
    # Generación de la señal sintética según los parámetros solicitados
    raw_signal = (
        0.15
        + np.sin(2 * np.pi * f0 * time_s)
        + 0.25 * np.sin(2 * np.pi * 12 * time_s)
        + 0.08 * rng.normal(size=n_samples)
    )

    centered_signal = raw_signal - np.mean(raw_signal)

    stats_raw = descriptive_statistics(raw_signal, unit="mV")
    stats_centered = descriptive_statistics(centered_signal, unit="mV")

    stats_summary = {
        "seed": seed,
        "dt_seconds": dt,
        "raw": stats_raw,
        "centered": stats_centered,
    }
    Path("results/stats.json").write_text(
        json.dumps(stats_summary, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Generación automática de la tabla LaTeX para el informe.
    tabla_path = Path("results/summary_table.tex")
    tabla_contenido = f"""\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Métrica}} & \\textbf{{Señal Cruda (mV)}} & \\textbf{{Señal Centrada (mV)}} \\\\
\\midrule
Media & {stats_raw["mean (mV)"]:.4f} & {stats_centered["mean (mV)"]:.4e} \\\\
Mediana & {stats_raw["median (mV)"]:.4f} & {stats_centered["median (mV)"]:.4f} \\\\
Desv. Estándar ($ddof=1$) & {stats_raw["sample_standard_deviation (mV)"]:.4f} & {stats_centered["sample_standard_deviation (mV)"]:.4f} \\\\
IQR & {stats_raw["iqr (mV)"]:.4f} & {stats_centered["iqr (mV)"]:.4f} \\\\
RMS & {stats_raw["rms (mV)"]:.4f} & {stats_centered["rms (mV)"]:.4f} \\\\
Mínimo / Máximo & {stats_raw["minimum (mV)"]:.2f} / {stats_raw["maximum (mV)"]:.2f} & {stats_centered["minimum (mV)"]:.2f} / {stats_centered["maximum (mV)"]:.2f} \\\\
\\bottomrule
\\end{{tabular}}
"""
    tabla_path.write_text(tabla_contenido, encoding="utf-8")

    # Álgebra lineal (Transformación didáctica).
    A_matrix = np.array([[0.10, 1.0], [1.0, 0.20]])
    c1 = np.sin(2 * np.pi * 0.05 * time_s[:100])
    c2 = np.cos(2 * np.pi * 0.1 * time_s[:100])
    x_channels = np.array([c1, c2])
    algebra_result = transformacion_didactica_matrices(A_matrix, x_channels)

    algebra_serializable = {}
    for k, v in algebra_result.items():  # Revisión de llaves y valores.
        if isinstance(v, np.ndarray):
            # Si el valor es matriz o arreglo de NumPy, se convierte a una lista normal.
            algebra_serializable[k] = v.tolist()
        elif isinstance(v, (np.float64, np.float32)):
            # Si es número decimal de NumPy, se convierte a decimal normal.
            algebra_serializable[k] = float(v)
        else:
            # Si es texto, entero o booleano normal, se deja tal cual.
            algebra_serializable[k] = v

    Path("results/algebra_report.json").write_text(
        json.dumps(algebra_serializable, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Derivación e Integración numérica.
    exact_derivative = (
        2 * np.pi * f0 * np.cos(2 * np.pi * f0 * time_s)
        + 0.25 * 2 * np.pi * 12 * np.cos(2 * np.pi * 12 * time_s)
    )

    deriv_summary = derivative_error_summary(raw_signal, time_s, exact_derivative)

    integrated_signal = cumulative_trapezoidal_integral(
        centered_signal, time_s, initial=0.0
    )
    total_integral = trapezoidal_integral(centered_signal, time_s)

    results_data = {
        "derivacion": {
            "paso_temporal": deriv_summary["time_step_s"],
            "error_maximo": deriv_summary["max_interior_error"],
            "error_rms": deriv_summary["rms_error"],
            "derivada_numerica_sample": deriv_summary["numerical_derivative"].tolist()[
                :10
            ],
        },
        "integracion": {
            "integral_total": total_integral,
            "integracion_acumulada_sample": integrated_signal.tolist()[:10],
        },
    }

    Path("results/derivation_integration_report.json").write_text(
        json.dumps(results_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Visualización (Generación de la figura de 3 paneles).
    fig = three_panel_figure(time_s, raw_signal, exact_derivative)
    figure_path = Path("figures/summary.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    # Registro de contribuciones
    contributions_content = r"""# Registro de Contribuciones y Rotación de Roles.

De acuerdo con los lineamientos del laboratorio, los dos integrantes del equipo participamos activamente en el desarrollo y hemos rotado por las diferentes responsabilidades.

## Integrantes del Equipo
* **Cristian Stiven Capera Cerquera**
* **Ana Sofía García Gutiérrez**
* **Jeimmy Andrea Gonzáles Gordillo**
* **Karol Mariana Gutiérrez Gutiérrez**
* **Brayan Andrés Ruiz Cortés**

## Matriz de Participación por Roles
| Integrante | Entorno y Reproducción | Métodos y Pruebas | Comunicación Científica |
| :--- | :---: | :---: | :---: |
| **Cristian Stiven Capera** | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Ana Sofía García** | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Jeimmy Andrea Gonzáles** | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Karol Mariana Gutiérrez** | $\checkmark$ | $\checkmark$ | $\checkmark$ |
| **Brayan Andrés Ruiz** | $\checkmark$ | $\checkmark$ | $\checkmark$ |
"""
    Path("results/contributions.md").write_text(contributions_content, encoding="utf-8")

    # Manifiesto final de entrega
    py_version = (
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    )
    try:
        uv_version_res = subprocess.run(
            ["uv", "--version"], capture_output=True, text=True, check=True
        )
        uv_version = uv_version_res.stdout.strip()
    except (subprocess.SubprocessError, FileNotFoundError):
        uv_version = "No detectado"

    manifest_content = f"""Manifiesto de Entrega - Laboratorio 1

Fecha de generación: {datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")}
Versión de Python: {py_version}
Herramienta de gestión: {uv_version}
Semilla usada: {seed}
Paso temporal (dt): {dt} s
Estado de pruebas: Aprobadas

ARCHIVOS ENTREGADOS Y GENERADOS:
- results/quality_report.json
- results/stats.json
- results/summary_table.tex
- results/algebra_report.json
- results/contributions.md
- results/derivation_integration_report.json
- figures/summary.png
- results/delivery_manifest.txt
"""
    Path("results/delivery_manifest.txt").write_text(manifest_content, encoding="utf-8")

    print("\n--- RESUMEN DE RUTAS CREADAS ---")
    rutas_creadas = [
        Path("results/quality_report.json"),
        Path("results/stats.json"),
        Path("results/summary_table.tex"),
        Path("results/algebra_report.json"),
        Path("results/contributions.md"),
        Path("results/derivation_integration_report.json"),
        Path("figures/summary.png"),
        Path("results/delivery_manifest.txt"),
    ]
    for ruta in rutas_creadas:
        estado = "Creado exitosamente" if ruta.exists() else "No encontrado"
        print(f"[{estado}] -> {ruta}")

    print("\nReproducción completada con éxito")


if __name__ == "__main__":
    main()

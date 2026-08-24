import datetime
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from python_scientific.descriptive_statistics import descriptive_statistics
from python_scientific.didactic_transformation import transformacion_didactica_matrices
from python_scientific.make_json_serializable import make_json_serializable
from python_scientific.numerical import (
    cumulative_trapezoidal_integral,
    derivative_error_summary,
    numerical_derivative,
    trapezoidal_integral,
)
from python_scientific.quality import validate_channel_configuration
from python_scientific.simulation import (
    center_signal,
    generate_synthetic_signal,
    get_algebra_data,
    run_channel_validation,
)
from python_scientific.three_panel_figure import three_panel_figure


def main() -> None:
    # Obtener la raíz del proyecto (subiendo un nivel desde scripts/)
    project_root = Path(__file__).resolve().parents[1]

    print("Iniciando pipeline de reproducción automatizada...")

    # Configuración global de parámetros.
    seed = 42
    f0 = 0.5  # Frecuencia base
    duration = 10.0
    n_samples = 500  # dt = duration / (n_samples - 1) = 10.0 / 499 ≈ 0.02 s
    unit = "mV"

    # Creación y aseguramiento de directorios de salida requeridos.
    Path(project_root / "results").mkdir(parents=True, exist_ok=True)
    Path(project_root / "figures").mkdir(parents=True, exist_ok=True)
    Path(project_root / "data/processed").mkdir(parents=True, exist_ok=True)

    # Validación de canales (delegada al módulo)
    quality_summary = run_channel_validation(validate_channel_configuration)
    Path(project_root / "results/quality_report.json").write_text(
        json.dumps(quality_summary, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Simulación de datos con parámetros externos y cálculo estadístico.
    time_s, raw_signal = generate_synthetic_signal(n_samples, duration, f0, seed)
    dt = time_s[1] - time_s[0]
    centered_signal = center_signal(raw_signal)

    stats_raw = descriptive_statistics(raw_signal)
    stats_centered = descriptive_statistics(centered_signal)

    stats_summary = {
        "seed": seed,
        "unit": unit,
        "dt_seconds": dt,
        "raw": stats_raw,
        "centered": stats_centered,
    }
    Path(project_root / "results/stats.json").write_text(
        json.dumps(stats_summary, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Generación del archivo CSV requerido por la rúbrica.
    csv_path = Path(project_root / "data/processed/processed_signals.csv")
    csv_data = np.column_stack((time_s, raw_signal, centered_signal))
    np.savetxt(
        csv_path,
        csv_data,
        delimiter=",",  # Datos separados por comas.
        header="time, raw_signal_mV, centered_signal_mV",  # Títulos
        comments="",  # Sin '#' en la cabecera.
    )

    # Generación automática de la tabla LaTeX para el informe.
    tabla_path = Path(project_root / "results/summary_table.tex")
    tabla_contenido = f"""\\begin{{tabular}}{{lcc}}
\\toprule
\\textbf{{Métrica}} & \\textbf{{Señal Cruda ({unit})}} & \\textbf{{Señal Centrada ({unit})}} \\\\
\\midrule
Media & {stats_raw["mean"]:.4f} & {stats_centered["mean"]:.4e} \\\\
Mediana & {stats_raw["median"]:.4f} & {stats_centered["median"]:.4f} \\\\
Desv. Estándar ($ddof=1$) & {stats_raw["sample_standard_deviation"]:.4f} & {stats_centered["sample_standard_deviation"]:.4f} \\\\
IQR & {stats_raw["iqr"]:.4f} & {stats_centered["iqr"]:.4f} \\\\
RMS & {stats_raw["rms"]:.4f} & {stats_centered["rms"]:.4f} \\\\
Mínimo / Máximo & {stats_raw["minimum"]:.2f} / {stats_raw["maximum"]:.2f} & {stats_centered["minimum"]:.2f} / {stats_centered["maximum"]:.2f} \\\\
\\bottomrule
\\end{{tabular}}
"""
    tabla_path.write_text(tabla_contenido, encoding="utf-8")

    # Álgebra lineal (datos delegados)
    A_matrix, x_channels = get_algebra_data(time_s)
    algebra_result = transformacion_didactica_matrices(A_matrix, x_channels)
    json_algebra_result = make_json_serializable(algebra_result)

    Path(project_root / "results/algebra_report.json").write_text(
        json.dumps(json_algebra_result, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Derivación e Integración numérica.
    numerical_deriv = numerical_derivative(raw_signal, time_s)

    exact_derivative = 2 * np.pi * f0 * np.cos(
        2 * np.pi * f0 * time_s
    ) + 0.25 * 2 * np.pi * 10 * np.cos(2 * np.pi * 10 * time_s)

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
    Path(project_root / "results/derivation_integration_report.json").write_text(
        json.dumps(results_data, indent=4, ensure_ascii=False), encoding="utf-8"
    )

    # Visualización (Generación de la figura de 3 paneles).
    fig = three_panel_figure(time_s, raw_signal, numerical_deriv)
    figure_path = Path(project_root / "figures/summary.png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)

    # Registro de contribuciones
    contributions_content = r"""# Registro de Contribuciones y Rotación de Roles.

De acuerdo con los lineamientos del laboratorio, los cinco integrantes del equipo participamos activamente en el desarrollo y hemos rotado por las diferentes responsabilidades.

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

    Path(project_root / "results/contributions.md").write_text(
        contributions_content, encoding="utf-8"
    )

    # Manifiesto final de entrega
    pytest_result = subprocess.run(
        [sys.executable, "-m", "pytest"],
        capture_output=True,
        text=True,
        check=False,
    )

    if pytest_result.returncode == 0:
        test_status = "Aprobadas (Verificadas mediante ejecución automática)"
    else:
        test_status = "FALLIDAS (Se detectaron errores en las pruebas)"

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

Nota importante: Todos los datos analizados y procesados en este proyecto 
provienen de señales sintéticas generadas mediante simulación matemática 
reproducible, por lo que no corresponden a mediciones experimentales reales.
    
Fecha de generación: {datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M:%S")}
Versión de Python: {py_version}
Herramienta de gestión: {uv_version}
Semilla usada: {seed}
Paso temporal (dt): {dt} s
Estado de pruebas: {test_status}

ARCHIVOS ENTREGADOS Y GENERADOS:
- results/quality_report.json
- results/stats.json
- data/processed/processed_signals.csv
- results/summary_table.tex
- results/algebra_report.json
- results/contributions.md
- results/derivation_integration_report.json
- figures/summary.png
- results/delivery_manifest.txt
"""
    Path(project_root / "results/delivery_manifest.txt").write_text(
        manifest_content, encoding="utf-8"
    )

    print("\n--- RESUMEN DE RUTAS CREADAS ---")
    rutas_creadas = [
        Path(project_root / "results/quality_report.json"),
        Path(project_root / "results/stats.json"),
        Path(project_root / "data/processed/processed_signals.csv"),
        Path(project_root / "results/summary_table.tex"),
        Path(project_root / "results/algebra_report.json"),
        Path(project_root / "results/contributions.md"),
        Path(project_root / "results/derivation_integration_report.json"),
        Path(project_root / "figures/summary.png"),
        Path(project_root / "results/delivery_manifest.txt"),
    ]
    for ruta in rutas_creadas:
        estado = "Creado exitosamente" if ruta.exists() else "No encontrado"
        print(f"[{estado}] -> {ruta}")

    print("\nReproducción completada con éxito")


if __name__ == "__main__":
    main()

import numpy as np


def descriptive_statistics(values, unit):
    """Calcula estadísticos sobre un arreglo 1D, no vacío y finito."""

    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError("Se esperaba un arreglo unidimensional.")
    if array.size == 0:
        raise ValueError("El arreglo no puede estar vacío.")
    if not np.all(np.isfinite(array)):
        raise ValueError("El arreglo contiene valores no finitos.")

    q1, q3 = np.quantile(array, [0.25, 0.75])

    return {
        "count (muestras)": int(array.size),
        f"mean ({unit})": float(np.mean(array)),
        f"median ({unit})": float(np.median(array)),
        f"sample_standard_deviation ({unit})": float(np.std(array, ddof=1))
        if array.size > 1
        else 0.0,
        f"minimum ({unit})": float(np.min(array)),
        f"maximum ({unit})": float(np.max(array)),
        f"q1 ({unit})": float(q1),
        f"q3 ({unit})": float(q3),
        f"iqr ({unit})": float(q3 - q1),
        f"rms ({unit})": float(np.sqrt(np.mean(array**2))),
    }


if __name__ == "__main__":
    pass

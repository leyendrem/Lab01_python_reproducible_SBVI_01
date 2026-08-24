import numpy as np


def descriptive_statistics(
    values: np.ndarray,
) -> dict[str, float]:
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
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "sample_standard_deviation": float(np.std(array, ddof=1))
        if array.size > 1
        else 0.0,
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
        "q1": float(q1),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        "rms": float(np.sqrt(np.mean(array**2))),
    }


if __name__ == "__main__":
    pass

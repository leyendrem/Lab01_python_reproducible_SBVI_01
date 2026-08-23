import numpy as np


def numerical_derivative(values, time_s):
    """Calcula la derivada numérica de una señal respecto al tiempo,
    pero antes se comprueba la estructura de las entradas."""

    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)

    if values.ndim != 1 or time_s.ndim != 1:
        raise ValueError("La señal y el tiempo deben ser arreglos unidimensionales.")
    if values.size != time_s.size:
        raise ValueError("La señal y el tiempo deben tener la misma longitud.")
    if values.size < 3:  # Para mirar hacia delante y hacia atrás.
        raise ValueError("Se necesitan al menos tres muestras.")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("El tiempo debe estar estrictamente ordenado.")

    return np.gradient(values, time_s)


def derivative_error_summary(values, time_s, exact_derivative):
    """Compara una derivada numérica con una derivada exacta,
    y se calculan errores máximo y RMS únicamente."""

    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)
    exact_derivative = np.asarray(exact_derivative, dtype=float)

    numerical = numerical_derivative(values, time_s)

    if exact_derivative.size != numerical.size:
        raise ValueError(
            "La derivada exacta y la derivada numérica deben tener la misma longitud."
        )

    interior_error = numerical[1:-1] - exact_derivative[1:-1]

    return {
        "time_step_s": float(np.mean(np.diff(time_s))),  # t[1] - t[0]
        "max_interior_error": float(np.max(np.abs(interior_error))),
        "rms_error": float(np.sqrt(np.mean(interior_error**2))),
        "numerical_derivative": numerical,
    }


def trapezoidal_integral(values, time_s):
    """Calcula una integral mediante la regla trapezoidal."""

    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)

    if values.ndim != 1 or time_s.ndim != 1:
        raise ValueError("La señal y el tiempo deben ser arreglos unidimensionales.")
    if values.size != time_s.size:
        raise ValueError("La señal y el tiempo deben tener la misma longitud.")
    if values.size < 2:  # Mínimo dos puntos en eje horizontal
        raise ValueError("Se necesitan al menos dos muestras.")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("El tiempo debe estar estrictamente ordenado.")

    return float(np.trapezoid(values, time_s))


def cumulative_trapezoidal_integral(values, time_s, initial=0.0):
    """Calcula la integral acumulada mediante la regla trapezoidal.
    La primera muestra comienza en el valor indicado por initial."""

    values = np.asarray(values, dtype=float)
    time_s = np.asarray(time_s, dtype=float)

    if values.ndim != 1 or time_s.ndim != 1:
        raise ValueError("La señal y el tiempo deben ser arreglos unidimensionales.")
    if values.size != time_s.size:
        raise ValueError("La señal y el tiempo deben tener la misma longitud.")
    if values.size < 2:
        raise ValueError("Se necesitan al menos dos muestras.")
    if np.any(np.diff(time_s) <= 0):
        raise ValueError("El tiempo debe estar estrictamente ordenado.")

    cumulative = np.zeros_like(values, dtype=float)
    cumulative[0] = initial

    increments = (values[:-1] + values[1:]) * np.diff(time_s) / 2.0

    cumulative[1:] = initial + np.cumsum(increments)

    return cumulative


if __name__ == "__main__":
    pass

import numpy as np


def transformacion_didactica_matrices(
    A: np.ndarray, x: np.ndarray
) -> dict[str, np.ndarray | float | bool]:
    """Realiza la transformación lineal, verificación
    y recuperación de canalesmsegún los requerimientos didácticos."""

    # Validación de la matriz A para ser cuadrada y 2D.
    A = np.asarray(A, dtype=float)
    x = np.asarray(x, dtype=float)

    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("La matriz A debe ser cuadrada (2D).")

    if A.shape != (2, 2):
        raise ValueError("La matriz A debe ser exactamente de 2x2.")

    y = A @ x
    if y.shape != x.shape:
        raise ValueError("Las formas de x e y deben coincidir.")

    # Se calcula el número de condición.
    cond_A = float(np.linalg.cond(A))

    # Se recuperan los canales mediante np.linalg.solve.
    x_recuperado = np.linalg.solve(A, y)

    # Se calcula el error RMS de reconstrucción.
    error_rms = float(np.sqrt(np.mean((x - x_recuperado) ** 2)))

    # Se hace una prueba con tolerancia explícita.
    tolerancia = 1e-7
    es_correcto = bool(np.allclose(x, x_recuperado, atol=tolerancia))

    if not es_correcto:
        raise ValueError("La reconstrucción excede la tolerancia permitida.")

    # Retorno del diccionario incluyendo la clave que pedía el test.
    return {
        "A": A,
        "x": x,
        "y": y,
        "cond_A": cond_A,
        "x_recuperado": x_recuperado,
        "error_rms": error_rms,
        "reconstruccion_exitosa": es_correcto,
    }


if __name__ == "__main__":
    pass

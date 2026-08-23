import numpy as np
import pytest

from python_scientific.didactic_transformation import transformacion_didactica_matrices


def test_didactic_transformation_success():

    # Matriz cuadrada de 2x2 bien condicionada
    A = np.array([[2.0, 1.0], [1.0, 3.0]])
    # Dos canales de prueba (señales 1D)
    x = np.array([[1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]])

    result = transformacion_didactica_matrices(A, x)

    # Verificación de que todas las llaves sean devueltas.
    assert "A" in result
    assert "x" in result
    assert "y" in result
    assert "cond_A" in result
    assert "x_recuperado" in result
    assert "error_rms" in result
    assert "reconstruccion_exitosa" in result

    # Verificación de la reconstrucción exitosa.
    assert result["reconstruccion_exitosa"] is True
    # El error RMS de recuperación debe ser prácticamente cero.
    assert result["error_rms"] < 1e-7
    # Las formas (shapes) deben mantenerse idénticas.
    assert result["x_recuperado"].shape == x.shape


def test_matrix_not_square():
    # Matriz que no es cuadrada (2x3).
    A = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        transformacion_didactica_matrices(A, x)


def test_matrix_wrong_size():
    # Matriz cuadrada pero de 3x3 en lugar de 2x2.
    A = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    x = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError):
        transformacion_didactica_matrices(A, x)

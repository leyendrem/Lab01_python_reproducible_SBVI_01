import numpy as np


def make_json_serializable(obj: object) -> object:
    """Convierte recursivamente arreglos de NumPy
    y tipos especiales a formatos serializables."""

    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [make_json_serializable(v) for v in obj]
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)

    return obj

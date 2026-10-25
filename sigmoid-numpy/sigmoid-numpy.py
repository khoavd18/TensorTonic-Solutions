import numpy as np

def sigmoid(x: list | float) -> np.ndarray | float:
    """
    Returns the sigmoid value for a scalar or each element of a list.
    """
    if isinstance(x, (int, float, np.number)):
        return float(1.0 / (1.0 + np.exp(-x)))

    arr = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-arr))
import numpy as np


# Used in other methods to generate numpy arrays in the shapes requested, with y taking values in [0, 1]
def generate_binary_data(X_shape, y_shape, y_two_dim=True):
    if y_two_dim:
        # Reshape y into a two-dimensional array
        if isinstance(y_shape, int): y_shape = (y_shape, 1)

    X = np.random.random(X_shape)
    y = np.random.choice([0, 1], size=y_shape)

    return X, y
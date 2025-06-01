import numpy as np

def load_cached_data():
    """
    Tries to load pre-processed data (.npy) from the current directory.
    """
    try:
        X_train = np.load('X_train.npy')
        y_train = np.load('y_train.npy')
        X_val   = np.load('X_val.npy')
        y_val   = np.load('y_val.npy')
        X_test  = np.load('X_test.npy')
        y_test  = np.load('y_test.npy')
        print("Data loaded from .npy files.")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except Exception as e:
        print(f"Could not load .npy data: {e}")
        return None
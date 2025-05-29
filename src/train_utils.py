import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
import logging

def flatten_images(X):
    """
    Converts images (N, H, W, C) into 1D vectors per image.
    """
    return X.reshape((X.shape[0], -1))

def get_svm_pipeline():
    """
    Returns a pipeline with PCA and SVM (configured).
    """
    pca = PCA(n_components=0.95, svd_solver='full')
    svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    return make_pipeline(pca, svm)

def get_rf_pipeline():
    """
    Returns a pipeline with PCA and Random Forest (configured).
    """
    pca = PCA(n_components=50)
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, random_state=42)
    return make_pipeline(pca, rf)

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
        logging.info("Data loaded from .npy files.")
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    except Exception as e:
        logging.warning(f"Could not load .npy data: {e}")
        return None
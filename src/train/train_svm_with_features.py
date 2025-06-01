import numpy as np
import joblib
import logging
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO)

def run_svm_training():
    logging.info("Loading extracted features...")
    X_train = np.load("features_X_train.npy")
    y_train = np.load("features_y_train.npy")

    logging.info(f"Shape: {X_train.shape}, Labels: {np.unique(y_train)}")

    logging.info("Training PCA + SVM pipeline...")
    pca = PCA(n_components=0.95, svd_solver='full')
    svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    model = make_pipeline(pca, svm)

    model.fit(X_train, y_train)

    joblib.dump(model, "svm_model.pkl")
    logging.info("SVM model saved to svm_model.pkl")
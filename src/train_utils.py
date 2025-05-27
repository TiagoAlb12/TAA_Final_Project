import numpy as np
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline

def flatten_images(X):
    """
    Converte imagens (N, H, W, C) em vetores 1D por imagem.
    """
    return X.reshape((X.shape[0], -1))

def get_svm_pipeline():
    """
    Retorna um pipeline com PCA + SVM (configurado).
    """
    pca = PCA(n_components=0.95, svd_solver='full')
    svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    return make_pipeline(pca, svm)

def get_rf_pipeline():
    """
    Retorna um pipeline com PCA + Random Forest (configurado).
    """
    pca = PCA(n_components=0.95, svd_solver='full')
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    return make_pipeline(pca, rf)

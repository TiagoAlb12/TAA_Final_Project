import os
import numpy as np
import joblib
import logging
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

logging.basicConfig(level=logging.INFO)

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    logging.info("Carregando features extraídas...")
    X_train = np.load(os.path.join(project_root, "X_train.npy"))
    y_train = np.load(os.path.join(project_root, "y_train.npy"))

    logging.info(f"Shape: {X_train.shape}, Labels: {np.unique(y_train)}")

    logging.info("Criando pipeline PCA + SVM...")
    pca = PCA(n_components=0.95, svd_solver='full')
    svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    model = make_pipeline(pca, svm)

    logging.info("Treinando modelo SVM com features da ResNet18...")
    model.fit(X_train, y_train)

    model_path = os.path.join(project_root, "svm_resnet_model.pkl")
    joblib.dump(model, model_path)
    logging.info(f"[✓] Modelo SVM salvo como {model_path}")

if __name__ == "__main__":
    main()

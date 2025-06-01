import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

def run_svm_training(model_path="svm_model.pkl"):
    print("Loading extracted features...")
    X_train = np.load("features_X_train.npy")
    y_train = np.load("features_y_train.npy")

    print(f"Shape: {X_train.shape}, Labels: {np.unique(y_train)}")

    print("Training PCA + SVM pipeline...")
    pca = PCA(n_components=0.95, svd_solver='full')
    svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True)
    model = make_pipeline(pca, svm)

    with tqdm(total=1, desc="Training SVM") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    joblib.dump(model, model_path)
    print(f"SVM model saved to {model_path}")
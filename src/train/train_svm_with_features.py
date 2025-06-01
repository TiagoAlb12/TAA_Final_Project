import numpy as np
import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from tqdm import tqdm

def run_svm_training(model_path="svm_model.pkl", use_weighted_classes=None):
    print("Loading extracted features...")
    X_train = np.load("features_X_train.npy")
    y_train = np.load("features_y_train.npy")

    print(f"Shape: {X_train.shape}, Labels: {np.unique(y_train)}")

    if use_weighted_classes is None:
        print("Note: SVMs can benefit from class weighting when dealing with imbalanced datasets,")
        print("but performance is usually not significantly affected.")
        resp = input("Use weighted classes for SVM? (y/n) [default: n]: ").strip().lower()
        use_weighted_classes = (resp == 'y')

    class_weight = 'balanced' if use_weighted_classes else None

    print(f"Training PCA + SVM pipeline (weighted classes: {use_weighted_classes})...")
    pca = PCA(n_components=0.95, svd_solver='full')
    svm = SVC(kernel='rbf', C=10, gamma=0.01, probability=True, class_weight=class_weight)
    model = make_pipeline(pca, svm)

    with tqdm(total=1, desc="Training SVM") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    joblib.dump(model, model_path)
    print(f"SVM model saved to {model_path}")
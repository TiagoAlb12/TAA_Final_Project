import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from tqdm import tqdm

def run_rf_training(model_path="rf_model.pkl", use_pca=False):
    print("Loading extracted features...")
    X_train = np.load("features_X_train.npy")
    y_train = np.load("features_y_train.npy")

    print(f"Feature shape: {X_train.shape}")

    print("Creating Random Forest pipeline...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=None,
        random_state=42
    )

    if use_pca:
        print("Adding PCA to pipeline...")
        pca = PCA(n_components=0.99, svd_solver='full')
        model = make_pipeline(pca, rf)
    else:
        model = rf

    with tqdm(total=1, desc="Training Random Forest") as pbar:
        model.fit(X_train, y_train)
        pbar.update(1)

    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")
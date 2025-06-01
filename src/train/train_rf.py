import numpy as np
import joblib
import logging
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA

logging.basicConfig(level=logging.INFO)

def run_rf_training(model_path="rf_model.pkl", use_pca=False):
    logging.info("Loading extracted features...")
    X_train = np.load("features_X_train.npy")
    y_train = np.load("features_y_train.npy")

    logging.info(f"Feature shape: {X_train.shape}")

    logging.info("Creating Random Forest pipeline...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight='balanced',
        max_depth=None,
        random_state=42
    )

    if use_pca:
        logging.info("Adding PCA to pipeline...")
        pca = PCA(n_components=0.99, svd_solver='full')
        model = make_pipeline(pca, rf)
    else:
        model = rf

    logging.info("Training Random Forest model...")
    model.fit(X_train, y_train)

    joblib.dump(model, model_path)
    logging.info(f"[✓] Model saved to {model_path}")
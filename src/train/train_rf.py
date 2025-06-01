import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
from tqdm import tqdm

def run_rf_training(model_path="rf_model.pkl", use_weighted_classes=None):
    print("Loading extracted features...")
    X_train = np.load("features_X_train.npy")
    y_train = np.load("features_y_train.npy")

    print(f"Feature shape: {X_train.shape}")

    if use_weighted_classes is None:
        print("Important: Random Forests are more sensitive to class imbalance,")
        print("so using weighted classes is strongly recommended.")
        resp = input("Use weighted classes for Random Forest? (y/n) [default: y]: ").strip().lower()
        use_weighted_classes = (resp != 'n')  # Default is yes

    class_weight = 'balanced' if use_weighted_classes else None

    print(f"Creating Random Forest (weighted classes: {use_weighted_classes})...")
    rf = RandomForestClassifier(
        n_estimators=200,
        class_weight=class_weight,
        max_depth=None,
        random_state=42
    )

    with tqdm(total=1, desc="Training Random Forest") as pbar:
        rf.fit(X_train, y_train)
        pbar.update(1)

    joblib.dump(rf, model_path)
    print(f"Model saved to {model_path}")
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt


def run_rf_evaluation(model_path="rf_model.pkl", output_dir="results/rf"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading Random Forest model...")
    model = joblib.load(model_path)

    if not os.path.exists("features_X_test.npy") or not os.path.exists("features_y_test.npy"):
        raise FileNotFoundError("Test data files not found. Run feature extraction first.")

    print("Loading test data...")
    X_test = np.load("features_X_test.npy")
    y_test = np.load("features_y_test.npy")

    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    print("Making predictions...")
    y_pred = model.predict(X_test_flat)
    y_probs = model.predict_proba(X_test_flat)
    np.save(os.path.join(output_dir, "rf_probs.npy"), y_probs)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    # Print
    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # Save classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix as image
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def run_svm_evaluation(output_dir="results/svm"):
    os.makedirs(output_dir, exist_ok=True)

    print("Loading SVM model...")
    model = joblib.load("svm_model.pkl")

    print("Loading test data...")
    X_test = np.load("features_X_test.npy")
    y_test = np.load("features_y_test.npy")

    print("Making predictions...")
    y_pred = model.predict(X_test)

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
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
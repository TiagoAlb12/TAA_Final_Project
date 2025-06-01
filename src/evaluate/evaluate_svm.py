import numpy as np
import joblib
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def run_svm_evaluation(model_path="svm_model.pkl", output_dir="results/svm"):
    os.makedirs(output_dir, exist_ok=True)

    print(f"Loading SVM model from {model_path}...")
    model = joblib.load(model_path)

    print("Loading test data...")
    X_test = np.load("features_X_test.npy")
    y_test = np.load("features_y_test.npy")

    # Define class names
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

    print("Making predictions...")
    y_pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)
        np.save(os.path.join(output_dir, "svm_probs.npy"), y_probs)

    acc = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {acc:.4f}")

    report = classification_report(y_test, y_pred, target_names=class_names, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print("\nClassification Report:\n", report)
    print("\nConfusion Matrix:\n", cm)

    # Save classification report
    with open(os.path.join(output_dir, "classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix image
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("SVM Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()
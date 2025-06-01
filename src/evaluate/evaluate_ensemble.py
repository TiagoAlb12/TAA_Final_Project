import numpy as np
import os
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

def run_ensemble_evaluation(results_root="results"):
    print("\n--- Running Ensemble Evaluation ---")

    model_dirs = ["cnn", "svm", "rf"]
    weights = [0.65, 0.26, 0.09]
    probs_list = []
    used_weights = []

    # Load y_test from current directory
    y_test = None
    for fallback in ["y_test.npy", "features_y_test.npy"]:
        if os.path.exists(fallback):
            y_test = np.load(fallback)
            break

    if y_test is None:
        print("Could not find 'y_test.npy' or 'features_y_test.npy' in the current directory.")
        return

    for model_name, weight in zip(model_dirs, weights):
        model_path = os.path.join(results_root, model_name)
        probs_file = os.path.join(model_path, f"{model_name}_probs.npy")

        if not os.path.exists(probs_file):
            print(f"Skipping {model_name} (missing {probs_file})")
            continue

        probs = np.load(probs_file)

        if probs.shape[0] != len(y_test):
            print(f"Skipping {model_name} due to shape mismatch with y_test.")
            continue

        probs_list.append(probs)
        used_weights.append(weight)

    if not probs_list:
        print("No valid models found for ensemble.")
        return

    # Normalize weights if needed
    used_weights = np.array(used_weights)
    used_weights = used_weights / used_weights.sum()

    print("Performing weighted soft voting...")
    ensemble_probs = sum(w * p for w, p in zip(used_weights, probs_list))
    y_pred = np.argmax(ensemble_probs, axis=1)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, digits=4)
    cm = confusion_matrix(y_test, y_pred)

    print(f"\nEnsemble Accuracy: {acc:.4f}")
    print("\nClassification Report:\n", report)
    print("Confusion Matrix:\n", cm)

    output_dir = os.path.join(results_root, "ensemble")
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "ensemble_classification_report.txt"), "w") as f:
        f.write(report)

    # Save confusion matrix
    class_names = ["MildDemented", "ModerateDemented", "NonDemented", "VeryMildDemented"]
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Ensemble Confusion Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # Save predictions and probabilities
    np.save(os.path.join(output_dir, "y_pred.npy"), y_pred)
    np.save(os.path.join(output_dir, "probs.npy"), ensemble_probs)

    print(f"\nEnsemble evaluation saved to: {output_dir}")
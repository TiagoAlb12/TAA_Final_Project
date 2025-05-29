import numpy as np
import joblib
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import os

logging.basicConfig(level=logging.INFO)

def main():
    logging.info("Carregando modelo Random Forest...")
    model = joblib.load("rf_model.pkl")

    logging.info("Carregando dados de teste...")

    if not os.path.exists("X_test.npy") or not os.path.exists("y_test.npy"):
        raise FileNotFoundError("Os ficheiros 'X_test.npy' e 'y_test.npy' não foram encontrados. Corre primeiro o extract_features.py")

    X_test = np.load("X_test.npy")
    y_test = np.load("y_test.npy")

    if y_test.ndim > 1 and y_test.shape[1] > 1:
        y_test = np.argmax(y_test, axis=1)

    X_test_flat = X_test.reshape((X_test.shape[0], -1))

    logging.info("Fazendo previsões...")
    y_pred = model.predict(X_test_flat)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"[✓] Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão - Random Forest")
    plt.savefig("confusion_matrix_rf.png")
    plt.show()

if __name__ == "__main__":
    main()

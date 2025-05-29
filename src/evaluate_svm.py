import os
import numpy as np
import joblib
import logging
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO)

def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

    logging.info("Carregando modelo SVM...")
    model = joblib.load(os.path.join(project_root, "svm_resnet_model.pkl"))

    logging.info("Carregando dados de teste...")
    X_test = np.load(os.path.join(project_root, "X_test.npy"))
    y_test = np.load(os.path.join(project_root, "y_test.npy"))

    logging.info("Fazendo previsões...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    logging.info(f"Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4))

    print("\nMatriz de Confusão:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predito")
    plt.ylabel("Verdadeiro")
    plt.title("Matriz de Confusão - SVM")
    plt.savefig(os.path.join(project_root, "confusion_matrix_svm.png"))
    plt.show()

if __name__ == "__main__":
    main()

import os
import json
import time
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                             roc_curve, auc, precision_recall_curve)
from sklearn.preprocessing import label_binarize
import torch
from models import CNNModel
from train_utils import flatten_images

def evaluate_model(model_path, X_test, y_test, output_dir='results', model_type='cnn', device=None):
    """
    Avalia um modelo treinado e salva métricas, gráficos e tempo de inferência.

    Args:
        model_path: Caminho para o modelo salvo (.pt para CNN, .pkl para SVM/RF)
        X_test: Dados de teste
        y_test: Labels de teste (one-hot ou inteiros)
        output_dir: Diretório onde salvar os resultados
        model_type: 'cnn', 'svm' ou 'rf'
        device: 'cuda' ou 'cpu'
    """
    os.makedirs(output_dir, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Validação de extensão
    if model_type == 'cnn' and not model_path.endswith('.pt'):
        raise ValueError("Para CNN, o modelo deve estar em formato .pt")
    elif model_type in ['svm', 'rf'] and not model_path.endswith('.pkl'):
        raise ValueError(f"Para {model_type.upper()}, o modelo deve estar em formato .pkl")

    # Carregamento do modelo
    if model_type == 'cnn':
        model = CNNModel()
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()
    else:
        model = joblib.load(model_path)
        X_test = flatten_images(X_test)  # SVM e RF requerem flatten

    # Conversão de y_test (one-hot → int)
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
        n_classes = y_test.shape[1]
    else:
        y_test_labels = y_test
        n_classes = len(np.unique(y_test))

    class_names = ['CN', 'EMCI', 'LMCI', 'AD']
    y_test_bin = label_binarize(y_test_labels, classes=np.arange(n_classes))

    # Predição e tempo de inferência
    start = time.time()
    if model_type == 'cnn':
        # Garantir batch e channel
        if X_test.ndim == 3:
            X_test = np.expand_dims(X_test, 1)  # [N, 1, H, W]
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
        with torch.no_grad():
            y_pred_logits = model(X_test_tensor)
            y_pred = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
            y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
    end = time.time()

    avg_inf_time = (end - start) / len(X_test)
    with open(f'{output_dir}/{model_type}_inference_time.txt', 'w') as f:
        f.write(f"Tempo médio por imagem: {avg_inf_time:.4f} segundos\n")

    # Relatório de Classificação
    report = classification_report(
        y_test_labels, y_pred_labels, target_names=class_names, output_dict=True)

    with open(f'{output_dir}/{model_type}_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    # Matrizes de Confusão
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {model_type.upper()}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_type}_confusion_matrix.png')
    plt.close()

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Matriz Normalizada - {model_type.upper()}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_type}_confusion_matrix_normalized.png')
    plt.close()

    # Curvas ROC
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8, 6))
    colors = ['blue', 'green', 'red', 'purple']
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label=f'{class_names[i]} (AUC = {roc_auc[i]:.2f})')
    plt.plot([0, 1], [0, 1], 'k--', lw=1)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taxa de Falsos Positivos')
    plt.ylabel('Taxa de Verdadeiros Positivos')
    plt.title(f'Curvas ROC - {model_type.upper()}')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_type}_roc_curves.png')
    plt.close()

    # Curvas Precision-Recall
    precision, recall, pr_auc = {}, {}, {}
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
        pr_auc[i] = auc(recall[i], precision[i])

    plt.figure(figsize=(8, 6))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color,
                 label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'Curvas Precision-Recall - {model_type.upper()}')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/{model_type}_precision_recall.png')
    plt.close()

    # Resumo de métricas
    metrics_summary = {
        'accuracy': report['accuracy'],
        'inference_time_avg': round(avg_inf_time, 4),
        'weighted_avg': {
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1_score': report['weighted avg']['f1-score']
        },
        'class_metrics': {
            class_names[i]: {
                'precision': report[class_names[i]]['precision'],
                'recall': report[class_names[i]]['recall'],
                'f1_score': report[class_names[i]]['f1-score'],
                'support': report[class_names[i]]['support']
            } for i in range(n_classes)
        }
    }

    with open(f'{output_dir}/{model_type}_metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"[✓] Avaliação concluída para modelo '{model_type.upper()}'. Resultados guardados em: {output_dir}/")

def load_test_data(data_dir):
    """
    Carrega dados de teste a partir de ficheiros .npy
    """
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    return X_test, y_test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Caminho para o modelo salvo')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Diretório com dados de teste')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['cnn', 'svm', 'rf'],
                        help='Tipo de modelo: cnn, svm ou rf')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Diretório para salvar os resultados')

    args = parser.parse_args()

    X_test, y_test = load_test_data(args.test_data)
    evaluate_model(
        model_path=args.model_path,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (classification_report, confusion_matrix, 
                            roc_curve, auc, precision_recall_curve)
from keras.models import load_model
import json
import os
from sklearn.preprocessing import label_binarize

def evaluate_model(model_path, X_test, y_test, output_dir='results', model_type='cnn'):
    """
    Avalia um modelo e salva métricas e visualizações
    
    Args:
        model_path: caminho para o modelo salvo (.h5 para CNN, .pkl para outros)
        X_test: dados de teste
        y_test: labels de teste (one-hot encoded)
        output_dir: diretório para salvar resultados
        model_type: 'cnn', 'svm' ou 'rf'
    """
    os.makedirs(output_dir, exist_ok=True)
    
    if model_type == 'cnn':
        model = load_model(model_path)
    else:
        import joblib
        model = joblib.load(model_path)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_labels = np.argmax(y_test, axis=1)
        n_classes = y_test.shape[1]
    else:
        y_test_labels = y_test
        n_classes = len(np.unique(y_test))
    
    if model_type == 'cnn':
        y_pred = model.predict(X_test)
        y_pred_labels = np.argmax(y_pred, axis=1)
    else:
        y_pred_labels = model.predict(X_test)
        y_pred = model.predict_proba(X_test)
    
    # 1. Relatório de Classificação
    class_names = ['CN', 'EMCI', 'LMCI', 'AD']
    report = classification_report(
        y_test_labels, 
        y_pred_labels, 
        target_names=class_names,
        output_dict=True
    )
    
    # Salvar relatório em JSON
    with open(f'{output_dir}/{model_type}_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)
    
    # 2. Matriz de Confusão
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test_labels, y_pred_labels)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, 
                yticklabels=class_names)
    plt.title(f'Matriz de Confusão - {model_type.upper()}')
    plt.ylabel('Verdadeiro')
    plt.xlabel('Predito')
    plt.savefig(f'{output_dir}/{model_type}_confusion_matrix.png')
    plt.close()
    
    # 3. Curvas ROC (apenas para CNN e modelos com predict_proba)
    if model_type != 'svm' or hasattr(model, 'predict_proba'):
        y_test_bin = label_binarize(y_test_labels, classes=np.arange(n_classes))
        
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_pred[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        
        plt.figure(figsize=(10, 8))
        colors = ['blue', 'green', 'red', 'purple']
        for i, color in zip(range(n_classes), colors):
            plt.plot(fpr[i], tpr[i], color=color,
                     label=f'ROC {class_names[i]} (AUC = {roc_auc[i]:.2f})')
        
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Taxa de Falso Positivo')
        plt.ylabel('Taxa de Verdadeiro Positivo')
        plt.title(f'Curvas ROC - {model_type.upper()}')
        plt.legend(loc="lower right")
        plt.savefig(f'{output_dir}/{model_type}_roc_curves.png')
        plt.close()
    
    # 4. Curvas Precision-Recall
    precision = dict()
    recall = dict()
    pr_auc = dict()
    
    for i in range(n_classes):
        precision[i], recall[i], _ = precision_recall_curve(y_test_bin[:, i], y_pred[:, i])
        pr_auc[i] = auc(recall[i], precision[i])
    
    plt.figure(figsize=(10, 8))
    for i, color in zip(range(n_classes), colors):
        plt.plot(recall[i], precision[i], color=color,
                 label=f'{class_names[i]} (AUC = {pr_auc[i]:.2f})')
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title(f'Curvas Precision-Recall - {model_type.upper()}')
    plt.legend(loc="lower left")
    plt.savefig(f'{output_dir}/{model_type}_precision_recall.png')
    plt.close()
    
    # 5. Salvar métricas importantes em um resumo
    metrics_summary = {
        'accuracy': report['accuracy'],
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
    
    print(f"Avaliação concluída. Resultados salvos em {output_dir}/")

def load_test_data(data_dir):
    """Carrega dados de teste pré-processados"""
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
                       help='Tipo de modelo a ser avaliado')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    X_test, y_test = load_test_data(args.test_data)
    
    evaluate_model(
        model_path=args.model_path,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.output_dir,
        model_type=args.model_type
    )
import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, precision_recall_curve
import torch
from ..models.resnet import get_resnet18
import torchvision.transforms as transforms

def evaluate_model(model_path, X_test, y_test, output_dir='results', device=None):
    """
    Evaluate a trained CNN model and save metrics, plots, and inference time.
    """
    os.makedirs(output_dir, exist_ok=True)
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the model
    model = get_resnet18(num_classes=4, grayscale=True)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Labels and classes
    y_test_labels = y_test
    n_classes = len(np.unique(y_test))
    class_names = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

    # Transformation for inference
    inference_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,), (0.229,))
    ])

    # Apply transform to each image and stack into a tensor
    print("Transforming test images...")
    start = time.time()
    X_test_transformed = []
    for i in range(len(X_test)):
        img = X_test[i]
        if img.ndim == 2:  # grayscale 2D
            img = np.expand_dims(img, -1)
        img_transformed = inference_transform(img)
        X_test_transformed.append(img_transformed)
    X_test_tensor = torch.stack(X_test_transformed).to(device)

    print("Running inference...")
    with torch.no_grad():
        y_pred_logits = model(X_test_tensor)
        y_pred = torch.softmax(y_pred_logits, dim=1).cpu().numpy()
        y_pred_labels = np.argmax(y_pred, axis=1)
    end = time.time()

    avg_inf_time = (end - start) / len(X_test)
    with open(f'{output_dir}/cnn_inference_time.txt', 'w') as f:
        f.write(f"Average inference time per image: {avg_inf_time:.4f} seconds\n")

    # Classification report
    print("Generating classification report...")
    report = classification_report(
        y_test_labels, y_pred_labels, target_names=class_names, output_dict=True, zero_division=0)

    with open(f'{output_dir}/cnn_classification_report.json', 'w') as f:
        json.dump(report, f, indent=4)

    # Confusion matrix
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_test_labels, y_pred_labels)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - CNN')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cnn_confusion_matrix.png')
    plt.close()

    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Greens',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Normalized Confusion Matrix - CNN')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cnn_confusion_matrix_normalized.png')
    plt.close()

    # ROC and Precision-Recall curves
    print("Generating ROC and Precision-Recall curves...")
    from sklearn.preprocessing import label_binarize
    y_test_bin = label_binarize(y_test_labels, classes=np.arange(n_classes))

    # ROC curves
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
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves - CNN')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cnn_roc_curves.png')
    plt.close()

    # Precision-Recall curves
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
    plt.title('Precision-Recall Curves - CNN')
    plt.legend(loc='lower left')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cnn_precision_recall.png')
    plt.close()

    # Metrics summary
    print("Saving metrics summary...")
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

    with open(f'{output_dir}/cnn_metrics_summary.json', 'w') as f:
        json.dump(metrics_summary, f, indent=4)

    print(f"Evaluation completed for CNN model. Results saved in: {output_dir}/")

def load_test_data(data_dir):
    """
    Load test data from .npy files.
    """
    X_test = np.load(f'{data_dir}/X_test.npy')
    y_test = np.load(f'{data_dir}/y_test.npy')
    return X_test, y_test

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to the saved model (.pt)')
    parser.add_argument('--test_data', type=str, required=True,
                        help='Directory containing test data')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save evaluation results')

    args = parser.parse_args()

    X_test, y_test = load_test_data(args.test_data)
    evaluate_model(
        model_path=args.model_path,
        X_test=X_test,
        y_test=y_test,
        output_dir=args.output_dir
    )
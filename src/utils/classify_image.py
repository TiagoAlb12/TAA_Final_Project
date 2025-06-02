import os
import numpy as np
import torch
import joblib
from torchvision import transforms
from PIL import Image
from src.utils.preprocessing import preprocess_image
from src.models.resnet import get_resnet18
from torchvision.models import resnet18
import torch.nn as nn

CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def classify_single_image(model_type, model_path, image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Pré-processamento comum
    img = preprocess_image(image_path)  # (224, 224, 1)
    if img is None:
        return
    img = img.astype(np.float32)

    print(f"Using model: {model_type.upper()}")

    if model_type == 'ensemble':
        return classify_single_image_ensemble(image_path, cnn_path=model_path[0], svm_path=model_path[1], rf_path=model_path[2])

    elif model_type == 'cnn':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.229,))
        ])
        img_tensor = transform(img.squeeze()).unsqueeze(0).to(device)

        model = get_resnet18(num_classes=4, grayscale=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)

    elif model_type in ('svm', 'rf'):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Transformar imagem para 3 canais (grayscale -> RGB)
        img_rgb = np.repeat(img, 3, axis=-1)  # (224, 224, 3)
        pil_img = Image.fromarray((img_rgb * 255).astype(np.uint8))

        transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        img_tensor = transform(pil_img).unsqueeze(0).to(device)

        # Modelo ResNet para extrair features
        feature_extractor = resnet18(weights='IMAGENET1K_V1')
        feature_extractor.fc = nn.Identity()
        feature_extractor.to(device)
        feature_extractor.eval()

        with torch.no_grad():
            features = feature_extractor(img_tensor).cpu().numpy()

        model = joblib.load(model_path)
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(features)[0]
        else:
            probs = np.zeros(len(CLASS_NAMES))
            probs[model.predict(features)[0]] = 1.0

        pred_idx = np.argmax(probs)

    else:
        print("Unsupported model type. Choose from: cnn, svm, rf.")
        return

    print(f"\n--- Prediction ---")
    print(f"Predicted class: {CLASS_NAMES[pred_idx]}")
    print(f"Probabilities: {dict(zip(CLASS_NAMES, np.round(probs, 4)))}")

    return CLASS_NAMES[pred_idx]

def classify_single_image_ensemble(image_path, cnn_path, svm_path, rf_path, weights=(0.40, 0.37, 0.23)):

    CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    img = preprocess_image(image_path)
    if img is None:
        return
    img = img.astype(np.float32)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # CNN
    transform_cnn = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485,), (0.229,))
    ])
    img_tensor_cnn = transform_cnn(img.squeeze()).unsqueeze(0).to(device)

    cnn_model = get_resnet18(num_classes=4, grayscale=True)
    cnn_model.load_state_dict(torch.load(cnn_path, map_location=device))
    cnn_model.to(device)
    cnn_model.eval()

    with torch.no_grad():
        cnn_output = cnn_model(img_tensor_cnn)
        cnn_probs = torch.softmax(cnn_output, dim=1).cpu().numpy()[0]

    # Features para SVM e RF
    img_rgb = np.repeat(img, 3, axis=-1)
    pil_img = Image.fromarray((img_rgb * 255).astype(np.uint8))
    img_tensor_feat = transforms.ToTensor()(pil_img).unsqueeze(0).to(device)

    feature_extractor = resnet18(weights='IMAGENET1K_V1')
    feature_extractor.fc = nn.Identity()
    feature_extractor.to(device)
    feature_extractor.eval()

    with torch.no_grad():
        features = feature_extractor(img_tensor_feat).cpu().numpy()

    svm_model = joblib.load(svm_path)
    rf_model = joblib.load(rf_path)

    svm_probs = svm_model.predict_proba(features)[0]
    rf_probs = rf_model.predict_proba(features)[0]

    # Aplicar pesos
    w_cnn, w_svm, w_rf = weights
    ensemble_probs = w_cnn * cnn_probs + w_svm * svm_probs + w_rf * rf_probs
    pred_idx = np.argmax(ensemble_probs)

    print(f"\n--- Weighted Ensemble Prediction ---")
    print(f"Model Weights → CNN: {w_cnn}, SVM: {w_svm}, RF: {w_rf}")
    print(f"Predicted class: {CLASS_NAMES[pred_idx]}")
    print(f"Probabilities: {dict(zip(CLASS_NAMES, np.round(ensemble_probs, 4)))}")

    return CLASS_NAMES[pred_idx]

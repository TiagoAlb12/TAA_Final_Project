import os
import numpy as np
import torch
import joblib
import cv2
from torchvision import transforms
from src.utils.preprocessing import preprocess_image
from src.models.resnet import get_resnet18

# Nome das classes
CLASS_NAMES = ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']

def classify_single_image(model_type, model_path, image_path):
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return

    # Pré-processamento comum
    img = preprocess_image(image_path)  # (224, 224, 1)
    if img is None:
        return
    img = img.astype(np.float32)  # <- força float32 para evitar incompatibilidade

    print(f"Using model: {model_type.upper()}")

    if model_type == 'cnn':
        # Converter imagem para tensor e normalizar como no treino
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485,), (0.229,))
        ])
        img_tensor = transform(img.squeeze()).unsqueeze(0).to(device)  # (1, 1, 224, 224)

        # Carregar modelo
        model = get_resnet18(num_classes=4, grayscale=True)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

        with torch.no_grad():
            output = model(img_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            pred_idx = np.argmax(probs)

    elif model_type in ('svm', 'rf'):
        # Flatten imagem para usar em modelos baseados em features
        flat_img = img.flatten().reshape(1, -1)
        model = joblib.load(model_path)

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(flat_img)[0]
        else:
            probs = np.zeros(len(CLASS_NAMES))
            probs[model.predict(flat_img)[0]] = 1.0

        pred_idx = np.argmax(probs)

    else:
        print("Unsupported model type. Choose from: cnn, svm, rf.")
        return

    print(f"\n--- Prediction ---")
    print(f"Predicted class: {CLASS_NAMES[pred_idx]}")
    print(f"Probabilities: {dict(zip(CLASS_NAMES, np.round(probs, 4)))}")

    return CLASS_NAMES[pred_idx]
import numpy as np
import cv2, os
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO)

def load_image_paths_and_labels(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"O diretório {data_dir} não foi encontrado.")
    if not os.listdir(data_dir):
        raise ValueError(f"O diretório {data_dir} está vazio.")

    image_paths = []
    labels = []
    class_names = sorted(os.listdir(data_dir))
    class_to_index = {class_name: idx for idx, class_name in enumerate(class_names)}

    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        if os.path.isdir(class_dir):
            for file_name in os.listdir(class_dir):
                if file_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_paths.append(os.path.join(class_dir, file_name))
                    labels.append(class_to_index[class_name])

    return image_paths, labels

def preprocess_image(path, target_size=(224, 224)):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Erro ao carregar a imagem: {path}")
        img = cv2.resize(img, target_size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = img / 255.0  # Normalização para [0, 1]
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        logging.error(f"Erro ao processar a imagem {path}: {e}")
        return None

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in image_paths:
        img = preprocess_image(path, target_size)
        if img is not None:
            images.append(img)
    return np.array(images)

def prepare_dataset(data_dir, test_size=0.15, val_size=0.15, num_classes=None, save_numpy=True):
    logging.info("Carregando caminhos das imagens e rótulos...")
    image_paths, labels = load_image_paths_and_labels(data_dir)

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), stratify=y_train_val)

    logging.info("Pré-processando imagens...")
    X_train = load_and_preprocess_images(X_train)
    X_val = load_and_preprocess_images(X_val)
    X_test = load_and_preprocess_images(X_test)

    # Convertendo os rótulos para arrays de inteiros (sem one-hot)
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    if save_numpy:
        logging.info("Salvando arrays pré-processados em .npy...")
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_val.npy", X_val)
        np.save("y_val.npy", y_val)
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)

    logging.info("Pré-processamento concluído.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# RF
def preprocess_image_rf(path, target_size=(64, 64)):
    """
    Pré-processa uma imagem para modelos tradicionais (Random Forest, SVM) com menor resolução.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Erro ao carregar a imagem: {path}")
        img = cv2.resize(img, target_size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        logging.error(f"Erro ao processar a imagem {path}: {e}")
        return None

def load_and_preprocess_images_rf(image_paths, target_size=(64, 64)):
    """
    Carrega e pré-processa imagens em baixa resolução (64x64).
    """
    images = []
    for path in image_paths:
        img = preprocess_image_rf(path, target_size)
        if img is not None:
            images.append(img)
    return np.array(images)

def prepare_dataset_rf(data_dir, test_size=0.15, val_size=0.15, save_numpy=True):
    """
    Prepara o dataset para Random Forest/SVM com imagens pequenas (64x64) e sem one-hot.
    """
    logging.info("Carregando caminhos das imagens e rótulos...")
    image_paths, labels = load_image_paths_and_labels(data_dir)
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), stratify=y_train_val)
    
    logging.info("Pré-processando imagens em baixa resolução...")
    X_train = load_and_preprocess_images_rf(X_train)
    X_val = load_and_preprocess_images_rf(X_val)
    X_test = load_and_preprocess_images_rf(X_test)

    if save_numpy:
        logging.info("Salvando arrays pequenos em .npy...")
        np.save("X_train_rf.npy", X_train)
        np.save("y_train_rf.npy", y_train)
        np.save("X_val_rf.npy", X_val)
        np.save("y_val_rf.npy", y_val)
        np.save("X_test_rf.npy", X_test)
        np.save("y_test_rf.npy", y_test)

    logging.info("Pré-processamento (small) concluído.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

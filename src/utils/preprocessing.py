import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def load_image_paths_and_labels(data_dir):
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"The directory {data_dir} was not found.")
    if not os.listdir(data_dir):
        raise ValueError(f"The directory {data_dir} is empty.")

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

    if len(image_paths) == 0:
        raise ValueError(f"No images found in directory '{data_dir}'. Make sure it contains valid images.")

    return image_paths, labels

def preprocess_image(path, target_size=(224, 224)):
    try:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Error loading image: {path}")
        img = cv2.resize(img, target_size)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        img = img / 255.0
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in tqdm(image_paths, desc="Preprocessing images"):
        img = preprocess_image(path, target_size)
        if img is not None:
            images.append(img)
    return np.array(images)


def prepare_dataset(data_dir, test_size=0.15, val_size=0.15, save_numpy=True):
    print("Loading image paths and labels...")
    image_paths, labels = load_image_paths_and_labels(data_dir)

    if len(image_paths) == 0 or len(set(labels)) < 2:
        raise ValueError("Not enough data to split. Ensure the dataset contains images from at least two classes.")

    X_train_val, X_test, y_train_val, y_test = train_test_split(
        image_paths, labels, test_size=test_size, stratify=labels)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=val_size / (1 - test_size), stratify=y_train_val)

    print("Preprocessing images...")
    X_train = load_and_preprocess_images(X_train)
    X_val = load_and_preprocess_images(X_val)
    X_test = load_and_preprocess_images(X_test)

    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)

    if save_numpy:
        print("Saving preprocessed arrays to .npy files...")
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", y_train)
        np.save("X_val.npy", X_val)
        np.save("y_val.npy", y_val)
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", y_test)

    print("Preprocessing completed.")
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)
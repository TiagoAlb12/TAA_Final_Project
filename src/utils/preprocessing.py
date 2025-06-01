import numpy as np
import cv2
import os
from tqdm import tqdm
from .train_utils import load_cached_data
from sklearn.model_selection import train_test_split
from PIL import Image
import imagehash
from collections import defaultdict
from random import shuffle

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
        img = img.astype(np.float16)
        img = np.expand_dims(img, axis=-1)
        return img
    except Exception as e:
        print(f"Error processing image {path}: {e}")
        return None

def group_duplicates_by_hash(image_paths, target_size=(224, 224)):
    print("Checking for duplicates using perceptual hashes...")
    hash_groups = defaultdict(list)
    for path in tqdm(image_paths, desc="Hashing images"):
        try:
            pil_img = Image.open(path).convert("L").resize(target_size)
            hash_val = str(imagehash.phash(pil_img))
            hash_groups[hash_val].append(path)
        except Exception as e:
            print(f"Hashing error for {path}: {e}")
    return list(hash_groups.values())

def split_hash_groups(groups, labels_dict, test_size=0.15, val_size=0.15):
    shuffle(groups)
    all_samples = [(path, labels_dict[path]) for group in groups for path in group]

    # Reconstruct labels for stratified splitting
    group_labels = [labels_dict[group[0]] for group in groups]

    train_val_groups, test_groups = train_test_split(groups, test_size=test_size, stratify=group_labels)
    train_val_labels = [labels_dict[group[0]] for group in train_val_groups]

    train_groups, val_groups = train_test_split(train_val_groups, test_size=val_size / (1 - test_size), stratify=train_val_labels)

    split_paths = lambda g: [p for group in g for p in group]
    return split_paths(train_groups), split_paths(val_groups), split_paths(test_groups)

def load_and_preprocess_images(image_paths, target_size=(224, 224)):
    images = []
    for path in tqdm(image_paths, desc="Preprocessing images"):
        img = preprocess_image(path, target_size)
        if img is not None:
            images.append(img)
    return np.array(images, dtype=np.float16)

def prepare_dataset(data_dir, test_size=0.15, val_size=0.15, save_numpy=True, force=False):
    if not force:
        cached = load_cached_data()
        if cached:
            use_cached = input("Cached dataset found. Reprocess images? (y/n): ").strip().lower()
            if use_cached != 'y':
                return cached
            else:
                print("Reprocessing dataset...")

    print("Loading image paths and labels...")
    image_paths, labels = load_image_paths_and_labels(data_dir)
    labels_dict = {path: label for path, label in zip(image_paths, labels)}

    hash_groups = group_duplicates_by_hash(image_paths)
    X_train_paths, X_val_paths, X_test_paths = split_hash_groups(hash_groups, labels_dict, test_size, val_size)

    y_train = [labels_dict[x] for x in X_train_paths]
    y_val = [labels_dict[x] for x in X_val_paths]
    y_test = [labels_dict[x] for x in X_test_paths]

    print("Preprocessing images...")

    X_train = load_and_preprocess_images(X_train_paths)
    X_val = load_and_preprocess_images(X_val_paths)
    X_test = load_and_preprocess_images(X_test_paths)

    if save_numpy:
        np.save("X_train.npy", X_train)
        np.save("y_train.npy", np.array(y_train))
        np.save("X_val.npy", X_val)
        np.save("y_val.npy", np.array(y_val))
        np.save("X_test.npy", X_test)
        np.save("y_test.npy", np.array(y_test))

    print("Preprocessing completed.")
    return (X_train, np.array(y_train)), (X_val, np.array(y_val)), (X_test, np.array(y_test))
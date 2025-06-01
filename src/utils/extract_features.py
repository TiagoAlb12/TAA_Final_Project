import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
from tqdm import tqdm

def extract_and_split_features(data_dir, test_ratio=0.2, batch_size=32):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(data_dir, transform=transform)

    if len(dataset) == 0:
        raise ValueError(f"No images found in directory '{data_dir}'.")
    if len(dataset.classes) < 2:
        raise ValueError("Dataset must contain at least two classes.")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    class_names = dataset.classes
    num_total = len(dataset)
    num_test = int(num_total * test_ratio)
    num_train = num_total - num_test

    train_data, test_data = random_split(dataset, [num_train, num_test])

    def extract(loader, desc=""):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in tqdm(loader, desc=desc):
                images = images.to(device)
                outputs = model(images)
                features.append(outputs.cpu().numpy())
                labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    X_train, y_train = extract(train_loader, desc="Extracting train features")
    X_test, y_test = extract(test_loader, desc="Extracting test features")

    np.save("features_X_train.npy", X_train)
    np.save("features_y_train.npy", y_train)
    np.save("features_X_test.npy", X_test)
    np.save("features_y_test.npy", y_test)

    print("Features extracted and saved.")
    print(f"Classes: {class_names}")

def run_feature_extraction(data_dir, force=False):
    required_files = [
        "features_X_train.npy",
        "features_y_train.npy",
        "features_X_test.npy",
        "features_y_test.npy"
    ]

    if not force and all(os.path.exists(f) for f in required_files):
        choice = input("Feature files already exist. Re-extract features? (y/n): ").strip().lower()
        if choice != 'y':
            print("Skipping feature extraction.")
            return

    print("Extracting features...")
    extract_and_split_features(data_dir)
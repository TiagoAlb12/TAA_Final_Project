import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import numpy as np
import os
import argparse

def extract_and_split_features(data_dir, test_ratio=0.2, batch_size=32):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Identity()
    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = ImageFolder(data_dir, transform=transform)
    class_names = dataset.classes
    num_total = len(dataset)
    num_test = int(num_total * test_ratio)
    num_train = num_total - num_test

    train_data, test_data = random_split(dataset, [num_train, num_test])

    def extract(loader):
        features, labels = [], []
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(device)
                outputs = model(images)
                features.append(outputs.cpu().numpy())
                labels.append(targets.numpy())
        return np.concatenate(features), np.concatenate(labels)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    print("[✓] Extraindo features de treino...")
    X_train, y_train = extract(train_loader)
    print("[✓] Extraindo features de teste...")
    X_test, y_test = extract(test_loader)

    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    np.save(os.path.join(root_dir, "X_train.npy"), X_train)
    np.save(os.path.join(root_dir, "y_train.npy"), y_train)
    np.save(os.path.join(root_dir, "X_test.npy"), X_test)
    np.save(os.path.join(root_dir, "y_test.npy"), y_test)

    print(f"[✓] Dados salvos em {root_dir}")
    print(f"[✓] Classes: {class_names}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--test_ratio', type=float, default=0.2)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    extract_and_split_features(args.data_dir, args.test_ratio, args.batch_size)

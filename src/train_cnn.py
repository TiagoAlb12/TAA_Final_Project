import logging
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from models import CNNModel
from preprocessing import prepare_dataset
from train_utils import load_cached_data
import numpy as np

logging.basicConfig(level=logging.INFO)

def train_cnn(data_dir, model_save_path, batch_size=32, epochs=100, patience=10, device=None):
    """
    Treina um modelo CNN com PyTorch.
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Usando dispositivo: {device}")

    cached = load_cached_data()
    if cached:
        (X_train, y_train), (X_val, y_val), _ = cached
    else:
        logging.info("[i] Dados não encontrados em cache. A processar imagens...")
        (X_train, y_train), (X_val, y_val), _ = prepare_dataset(data_dir, save_numpy=True)

    # Adiciona esta conversão para garantir float32
    X_train = X_train.astype(np.float32)
    X_val = X_val.astype(np.float32)

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Os dados de treinamento estão vazios.")
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("Os dados de validação estão vazios.")

    # Transformações (augmentação)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(15),
        transforms.RandomResizedCrop(224, scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ])

    # PyTorch datasets
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self, X, y, transform=None):
            self.X = X
            self.y = y
            self.transform = transform
        def __len__(self):
            return len(self.X)
        def __getitem__(self, idx):
            x = self.X[idx]
            y = self.y[idx]
            if self.transform:
                x = self.transform(x)
            else:
                x = torch.tensor(x, dtype=torch.float32)
            y = torch.tensor(y, dtype=torch.long)
            return x, y

    train_dataset = CustomDataset(X_train, y_train, transform=transform)
    val_dataset = CustomDataset(X_val, y_val, transform=transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    best_val_loss = float('inf')
    patience_counter = 0
    history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}

    logging.info("Iniciando o treino...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels.argmax(dim=1) if labels.ndim > 1 else labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            if labels.ndim > 1:
                labels_max = labels.argmax(dim=1)
            else:
                labels_max = labels
            correct += (predicted == labels_max).sum().item()
            total += inputs.size(0)
        train_loss = running_loss / total
        train_acc = correct / total

        # Validação
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels.argmax(dim=1) if labels.ndim > 1 else labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                if labels.ndim > 1:
                    labels_max = labels.argmax(dim=1)
                else:
                    labels_max = labels
                val_correct += (predicted == labels_max).sum().item()
                val_total += inputs.size(0)
        val_loss = val_loss / val_total
        val_acc = val_correct / val_total

        history['loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['accuracy'].append(train_acc)
        history['val_accuracy'].append(val_acc)

        logging.info(f"Epoch {epoch+1}/{epochs} - loss: {train_loss:.4f} - acc: {train_acc:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), model_save_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info("Early stopping!")
                break

    logging.info("Treino concluído. A guardar o histórico...")
    with open('training_history.json', 'w') as f:
        json.dump(history, f)

    logging.info(f"[✓] Modelo CNN guardado em: {model_save_path}")
    return history
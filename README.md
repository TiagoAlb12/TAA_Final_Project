Final Group Project TAA

# Dependencies

1. **Create a Virtual Environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

2. **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

3. **Install PyTorch:**
    - Refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for the correct command based on your system (and CUDA version).
    - Alternatively, check the `PyTorchInstalation.txt` file in the project directory for installation instructions.

4. **Install Additional Dependencies (if any):**
    - Follow any extra instructions provided in the `requirements.txt` or project documentation.

Make sure all dependencies are installed before running the project.

# CNN Algorithm:

*(Inside the venv)*

## 1. Prepare data (only needed once)
```bash
python3 src/main.py --task prepare --data_dir images/OriginalDataset
```

## 2. Train with ResNet18
```bash
python3 src/main.py --task train --model_type cnn --data_dir images/OriginalDataset --model_path cnn_resnet18.pth --epochs 30 --batch_size 32
```

## 3. Evaluate model
```bash
python3 src/main.py --task evaluate --model_type cnn --data_dir images/OriginalDataset --model_path cnn_resnet18.pth --output_dir results_resnet
```

# SVM Algorithm:

*(Specific script)*

## 1. Activate the venv

## 2. Go to src/utils

## 3. Run the following commands:
```bash
chmod +x run_svm.sh
./run_svm.sh
```

---
---

# Algoritmo RF

*(Script específico)*

### 1. Abrir o venv

### 2. Ir para src/utils

### 3. Executar os seguintes comando:
```bash
chmod +x run_rf.sh
./run_rf.sh
```
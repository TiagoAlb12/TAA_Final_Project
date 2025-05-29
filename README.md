# TAA_Final_Project
Projeto Final de Grupo TAA

---
---
# Algoritmo CNN:

*(Dentro do venv)*

### 1. Preparar dados (só é preciso 1x)
```bash
python3 src/main.py --task prepare --data_dir images/OriginalDataset
```

### 2. Treinar com ResNet18
```bash
python3 src/main.py --task train --model_type cnn --data_dir images/OriginalDataset --model_path cnn_resnet18.pth --epochs 30 --batch_size 32
```

### 3. Avaliar modelo
```bash
python3 src/main.py --task evaluate --model_type cnn --data_dir images/OriginalDataset --model_path cnn_resnet18.pth --output_dir results_resnet
```

---
---

# Algoritmo SVM:

*(Script específico)*

### 1. Abrir o venv

### 2. Ir para src/utils

### 3. Executar os seguintes comando:
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
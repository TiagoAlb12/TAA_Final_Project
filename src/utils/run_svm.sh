# SVM

echo "[1/3] Extraindo features e dividindo treino/teste..."
python3 ../extract_features.py --data_dir ../../images/OriginalDataset

echo "[2/3] Treinando modelo SVM com ResNet18 + PCA..."
python3 ../train_svm_with_features.py

echo "[3/3] Avaliando modelo SVM..."
python3 ../evaluate_svm.py

# Para rodar este script, executar:
# chmod +x run_svm.sh
# ./run_svm.sh

# NOTA: Executar com venv e dentro da pasta src/utils
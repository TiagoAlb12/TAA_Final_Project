# RF

echo "[1/3] Preparando dados (se necessário)..."
python3 ../extract_features.py --data_dir ../../images/OriginalDataset

echo "[2/3] Treinando modelo Random Forest..."
python3 ../train_rf.py --data_dir ../../images/OriginalDataset --model_path rf_model.pkl

echo "[3/3] Avaliando modelo Random Forest..."
python3 ../evaluate_rf.py

# Para rodar este script, executar:
# chmod +x run_rf.sh
# ./run_rf.sh
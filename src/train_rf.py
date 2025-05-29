import argparse
import logging
import joblib
import numpy as np
from preprocessing import prepare_dataset_rf
from train_utils import flatten_images, get_rf_pipeline, load_cached_data  # <- NOVO

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Treino de modelo Random Forest para classificação de Alzheimer")
    parser.add_argument('--data_dir', type=str, required=True, help="Diretório contendo o dataset")
    parser.add_argument('--model_path', type=str, default='rf_model.pkl', help="Caminho para salvar o modelo RF")
    args = parser.parse_args()

    logging.info("Carregando ou preparando dados de treino...")
    cached = load_cached_data()
    if cached:
        (X_train, y_train), _, _ = cached
    else:
        (X_train, y_train), _, _ = prepare_dataset_rf(args.data_dir, save_numpy=True)

    X_train_flat = flatten_images(X_train)
    y_train_class = np.argmax(y_train, axis=1)

    logging.info("Criando pipeline com PCA + Random Forest...")
    model = get_rf_pipeline()

    logging.info("Treinando Random Forest...")
    model.fit(X_train_flat, y_train_class)

    joblib.dump(model, args.model_path)
    logging.info(f"[✓] Modelo RF salvo em: {args.model_path}")

if __name__ == '__main__':
    main()

import argparse
import logging
import joblib
import numpy as np
from preprocessing import prepare_dataset
from train_cnn import train_cnn
from train_utils import flatten_images, get_svm_pipeline, get_rf_pipeline
from evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)

def train_model(args):
    if args.model_type == 'cnn':
        train_cnn(args.data_dir, args.model_path, batch_size=args.batch_size,
                  epochs=args.epochs, patience=args.patience)

    else:
        if args.use_resnet_features:
            logging.info("[i] Usando features extraídas por ResNet18 para treino do modelo...")
            X_train = np.load("X_train_features.npy")
            y_train = np.load("y_train.npy")
            X_train_flat = X_train
        else:
            logging.info("[i] Usando imagens cruas (flattened) para treino do modelo...")
            (X_train, y_train), _, _ = prepare_dataset(args.data_dir, save_numpy=True)
            X_train_flat = flatten_images(X_train)
            y_train = np.argmax(y_train, axis=1)

        model = get_svm_pipeline() if args.model_type == 'svm' else get_rf_pipeline()
        logging.info(f"Treinando modelo {args.model_type.upper()}...")
        model.fit(X_train_flat, y_train)

        joblib.dump(model, args.model_path)
        logging.info(f"[✓] Modelo salvo em: {args.model_path}")

def main():
    parser = argparse.ArgumentParser(description="Pipeline para classificação de Alzheimer")
    parser.add_argument('--data_dir', type=str, required=True, help="Diretório contendo o dataset")
    parser.add_argument('--model_path', type=str, default="cnn_model.h5", help="Caminho para salvar ou carregar o modelo")
    parser.add_argument('--output_dir', type=str, default="results", help="Diretório para salvar os resultados da avaliação")
    parser.add_argument('--task', type=str, required=True, choices=['prepare', 'train', 'evaluate', 'full'],
                        help="Tarefa a executar: prepare, train, evaluate ou full")
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'svm', 'rf'],
                        help="Tipo de modelo a usar: cnn, svm ou rf")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do batch (apenas CNN)")
    parser.add_argument('--epochs', type=int, default=10, help="Número de épocas (apenas CNN)")
    parser.add_argument('--patience', type=int, default=5, help="Patience para early stopping (apenas CNN)")
    parser.add_argument('--use_resnet_features', action='store_true', help="Usar features da ResNet18 para SVM/RF")

    args = parser.parse_args()

    if args.task == 'prepare':
        logging.info("Preparando o dataset...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir, save_numpy=True)
        logging.info("Dataset preparado com sucesso!")

    elif args.task == 'train':
        train_model(args)

    elif args.task == 'evaluate':
        if args.use_resnet_features:
            logging.info("Carregando features de teste...")
            X_test = np.load("X_test_features.npy")
            y_test = np.load("y_test.npy")
        else:
            logging.info("Carregando dados de teste para avaliação...")
            (_, _), (_, _), (X_test, y_test) = prepare_dataset(args.data_dir, save_numpy=True)

        evaluate_model(args.model_path, X_test, y_test, output_dir=args.output_dir, model_type=args.model_type)

    elif args.task == 'full':
        logging.info("Executando pipeline completo...")

        if args.use_resnet_features:
            logging.warning("[!] Full pipeline com ResNet18 requer extração de features fora do main.py")
            logging.info("Executa `extract_features.py` antes de usar este modo com `--use_resnet_features`.")
            return

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir, save_numpy=True)
        train_model(args)
        evaluate_model(args.model_path, X_test, y_test, output_dir=args.output_dir, model_type=args.model_type)
        logging.info("Pipeline completo executado com sucesso!")

if __name__ == '__main__':
    main()

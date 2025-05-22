import argparse
import logging
from src.preprocessing import prepare_dataset
from src.models import create_cnn_model
from src.train import train_cnn
from src.evaluate import evaluate_model

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Pipeline para classificação de Alzheimer")
    parser.add_argument('--data_dir', type=str, required=True, help="Diretório contendo o dataset")
    parser.add_argument('--model_path', type=str, default="cnn_model.h5", help="Caminho para salvar ou carregar o modelo")
    parser.add_argument('--output_dir', type=str, default="results", help="Diretório para salvar os resultados da avaliação")
    parser.add_argument('--task', type=str, required=True, choices=['prepare', 'train', 'evaluate', 'full'],
                        help="Tarefa a executar: prepare, train, evaluate ou full")
    parser.add_argument('--batch_size', type=int, default=32, help="Tamanho do batch para treinamento")
    parser.add_argument('--epochs', type=int, default=10, help="Número de épocas para treinamento")
    parser.add_argument('--patience', type=int, default=5, help="Número de épocas para early stopping")

    args = parser.parse_args()

    if args.task == 'prepare':
        logging.info("Preparando o dataset...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir)
        logging.info("Dataset preparado com sucesso!")

    elif args.task == 'train':
        logging.info("Iniciando o treinamento do modelo...")
        train_cnn(args.data_dir, args.model_path, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
        logging.info("Treinamento concluído!")

    elif args.task == 'evaluate':
        logging.info("Carregando dados de teste e avaliando o modelo...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir)
        evaluate_model(args.model_path, X_test, y_test, output_dir=args.output_dir)
        logging.info("Avaliação concluída!")

    elif args.task == 'full':
        logging.info("Executando o pipeline completo...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir)
        train_cnn(args.data_dir, args.model_path, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience)
        evaluate_model(args.model_path, X_test, y_test, output_dir=args.output_dir)
        logging.info("Pipeline completo executado com sucesso!")

if __name__ == '__main__':
    main()
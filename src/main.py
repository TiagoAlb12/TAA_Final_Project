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
            logging.info("Using features extracted by ResNet18 for model training...")
            X_train = np.load("X_train_features.npy")
            y_train = np.load("y_train.npy")
            X_train_flat = X_train
        else:
            logging.info("Using raw (flattened) images for model training...")
            (X_train, y_train), _, _ = prepare_dataset(args.data_dir, save_numpy=True)
            X_train_flat = flatten_images(X_train)
            y_train = np.argmax(y_train, axis=1)

        model = get_svm_pipeline() if args.model_type == 'svm' else get_rf_pipeline()
        logging.info(f"Training {args.model_type.upper()} model...")
        model.fit(X_train_flat, y_train)

        joblib.dump(model, args.model_path)
        logging.info(f"Model saved at: {args.model_path}")

def main():
    logging.info("Starting...")
    parser = argparse.ArgumentParser(description="Alzheimer classification pipeline")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the dataset")
    parser.add_argument('--model_path', type=str, default="cnn_model.h5", help="Path to save or load the model")
    parser.add_argument('--output_dir', type=str, default="results", help="Directory to save evaluation results")
    parser.add_argument('--task', type=str, required=True, choices=['prepare', 'train', 'evaluate', 'full'],
                        help="Task to execute: prepare, train, evaluate or full")
    parser.add_argument('--model_type', type=str, default='cnn', choices=['cnn', 'svm', 'rf'],
                        help="Model type to use: cnn, svm or rf")
    parser.add_argument('--batch_size', type=int, default=32, help="Batch size (CNN only)")
    parser.add_argument('--epochs', type=int, default=10, help="Number of epochs (CNN only)")
    parser.add_argument('--patience', type=int, default=5, help="Patience for early stopping (CNN only)")
    parser.add_argument('--use_resnet_features', action='store_true', help="Use ResNet18 features for SVM/RF")

    args = parser.parse_args()

    if args.task == 'prepare':
        logging.info("Preparing the dataset...")
        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir, save_numpy=True)
        logging.info("Dataset prepared successfully.")

    elif args.task == 'train':
        train_model(args)

    elif args.task == 'evaluate':
        if args.use_resnet_features:
            logging.info("Loading test features...")
            X_test = np.load("X_test_features.npy")
            y_test = np.load("y_test.npy")
        else:
            logging.info("Loading test data for evaluation...")
            (_, _), (_, _), (X_test, y_test) = prepare_dataset(args.data_dir, save_numpy=True)

        evaluate_model(args.model_path, X_test, y_test, output_dir=args.output_dir, model_type=args.model_type)

    elif args.task == 'full':
        logging.info("Running full pipeline...")

        if args.use_resnet_features:
            logging.warning("Full pipeline with ResNet18 requires feature extraction outside main.py")
            logging.info("Run `extract_features.py` before using this mode with `--use_resnet_features`.")
            logging.info("Finished.")
            return

        (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(args.data_dir, save_numpy=True)
        train_model(args)
        evaluate_model(args.model_path, X_test, y_test, output_dir=args.output_dir, model_type=args.model_type)
        logging.info("Full pipeline executed successfully.")

    logging.info("Finished.")

if __name__ == '__main__':
    main()
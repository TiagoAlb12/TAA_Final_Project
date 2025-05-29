import argparse
import logging
import joblib
import numpy as np
from preprocessing import prepare_dataset
from train_utils import flatten_images, get_svm_pipeline, load_cached_data

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser(description="Train SVM model for Alzheimer classification")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing the dataset")
    parser.add_argument('--model_path', type=str, default='svm_model.pkl', help="Path to save the SVM model")
    args = parser.parse_args()

    logging.info("Loading or preparing training data...")
    cached = load_cached_data()
    if cached:
        (X_train, y_train), _, _ = cached
    else:
        (X_train, y_train), _, _ = prepare_dataset(args.data_dir, save_numpy=True)

    X_train_flat = flatten_images(X_train)
    y_train_class = np.argmax(y_train, axis=1)

    logging.info("Creating pipeline with PCA and SVM...")
    model = get_svm_pipeline()

    logging.info("Training SVM...")
    model.fit(X_train_flat, y_train_class)

    joblib.dump(model, args.model_path)
    logging.info(f"Model saved at: {args.model_path}")

if __name__ == '__main__':
    main()

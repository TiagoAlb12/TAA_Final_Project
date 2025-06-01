from src.utils.preprocessing import prepare_dataset
from src.train.train_cnn import train_cnn
from src.evaluate.evaluate_cnn import evaluate_model
from src.utils.extract_features import run_feature_extraction
from src.train.train_svm_with_features import run_svm_training
from src.evaluate.evaluate_svm import run_svm_evaluation
from src.train.train_rf import run_rf_training
from src.evaluate.evaluate_rf import run_rf_evaluation
import os
import glob
import shutil

def train_model(model_type, data_dir, model_path, batch_size=32, epochs=10, patience=5):
    if model_type == 'cnn':
        train_cnn(data_dir, model_path, batch_size=batch_size, epochs=epochs, patience=patience)

def cleanup_generated_files(results_root="results"):
    extensions = ["*.npy", "*.pkl", "*.pth", "*.json", "*.png"]

    files_deleted = 0
    dirs_deleted = 0

    for ext in extensions:
        for file in glob.glob(ext):
            try:
                os.remove(file)
                print(f"Deleted file: {file}")
                files_deleted += 1
            except Exception as e:
                print(f"Could not delete file {file}: {e}")

    if os.path.isdir(results_root):
        try:
            shutil.rmtree(results_root)
            print(f"Deleted directory: {results_root}/")
            dirs_deleted += 1
        except Exception as e:
            print(f"Could not delete directory {results_root}: {e}")

    if files_deleted == 0 and dirs_deleted == 0:
        print("No generated files or directories found.")
    else:
        print(f"Cleanup complete. {files_deleted} files and {dirs_deleted} directories deleted.")

def menu():
    print("Alzheimer Classification Pipeline Ready")

    data_dir = input("Enter dataset directory path [default: ./images/OriginalDataset]: ").strip() or "./images/OriginalDataset"
    model_path = input("Enter path to save the model [default: cnn_resnet18.pth]: ").strip() or "cnn_resnet18.pth"
    results_root = input("Enter base results directory [default: results]: ").strip() or "results"

    while True:
        print("\n--- Main Menu ---")
        print("1. Prepare dataset")
        print("2. Train model")
        print("3. Evaluate model")
        print("4. Run full pipeline (CNN only)")
        print("5. Cleanup generated files")
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == '1':
            print("Preparing dataset...")
            prepare_dataset(data_dir, save_numpy=True)
            print("Dataset preparation complete.")

        elif choice == '2':
            model_type = input("Select model type (cnn / svm / rf): ").strip().lower()
            
            if model_type == 'cnn':
                batch_size = int(input("Batch size [default: 32]: ") or 32)
                epochs = int(input("Number of epochs [default: 10]: ") or 10)
                patience = int(input("Early stopping patience [default: 5]: ") or 5)
                train_model(model_type, data_dir, model_path, batch_size, epochs, patience)

            elif model_type == 'svm':
                run_feature_extraction(data_dir)
                print("Training SVM model...")
                run_svm_training()

            elif model_type == 'rf':
                run_feature_extraction(data_dir)
                print("Training Random Forest model...")
                run_rf_training()

            else:
                print("Unsupported model type.")

        elif choice == '3':
            model_type = input("Model type to evaluate (cnn / svm / rf): ").strip().lower()
            output_dir_custom = input(f"Enter subdirectory name for this model [default: {model_type}]: ").strip()
            subfolder = output_dir_custom or model_type
            output_dir_eval = os.path.join(results_root, subfolder)

            if model_type == 'cnn':
                (_, _), (_, _), (X_test, y_test) = prepare_dataset(data_dir, save_numpy=True)
                evaluate_model(model_path, X_test, y_test, output_dir=output_dir_eval)

            elif model_type == 'svm':
                run_svm_evaluation(output_dir=output_dir_eval)

            elif model_type == 'rf':
                run_rf_evaluation(output_dir=output_dir_eval)

            else:
                print("Unsupported model type.")

        elif choice == '4':
            print("Running full CNN pipeline...")
            (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(data_dir, save_numpy=True)
            train_model('cnn', data_dir, model_path)
            evaluate_model(model_path, X_test, y_test, output_dir=results_root)
            print("Full pipeline complete.")

        elif choice == '5':
            print("Cleaning up generated files...")
            cleanup_generated_files(results_root)

        elif choice == '0':
            print("Exiting.")
            break
        else:
            print("Invalid option. Please try again.")

if __name__ == '__main__':
    print("Starting Alzheimer Classification Pipeline...")
    menu()
from src.utils.preprocessing import prepare_dataset
from src.train.train_cnn import train_cnn
from src.evaluate.evaluate_cnn import evaluate_model
from src.utils.extract_features import run_feature_extraction
from src.train.train_svm_with_features import run_svm_training
from src.evaluate.evaluate_svm import run_svm_evaluation
from src.train.train_rf import run_rf_training
from src.evaluate.evaluate_rf import run_rf_evaluation
from src.evaluate.evaluate_ensemble import run_ensemble_evaluation
import os
import glob
import shutil

def train_model(model_type, data_dir, model_path_cnn, batch_size=32, epochs=30, patience=5):
    if model_type == 'cnn':
        train_cnn(data_dir, model_path_cnn, batch_size=batch_size, epochs=epochs, patience=patience)

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

def setup_paths():
    print("\n--- Setup Configuration ---")
    data_dir = input("Enter dataset directory path [default: ./images/OriginalDataset]: ").strip() or "./images/OriginalDataset"
    while not os.path.isdir(data_dir):
        print(f"Error: Provided dataset directory '{data_dir}' does not exist.")
        data_dir = input("Enter dataset directory path [default: ./images/OriginalDataset]: ").strip() or "./images/OriginalDataset"
    
    model_path_cnn = input("Enter path to save the CNN model [default: cnn_resnet18.pth]: ").strip() or "cnn_resnet18.pth"
    model_path_svm = input("Enter path to save the SVM model [default: svm_model.pkl]: ").strip() or "svm_model.pkl"
    model_path_rf  = input("Enter path to save the RF model [default: rf_model.pkl]: ").strip() or "rf_model.pkl"
    results_root = input("Enter base results directory [default: results]: ").strip() or "results"

    return data_dir, model_path_cnn, model_path_svm, model_path_rf, results_root

def menu():
    print("Alzheimer Classification Pipeline Ready")

    data_dir, model_path_cnn, model_path_svm, model_path_rf, results_root = setup_paths()

    while True:
        print("\n--- Main Menu ---")
        print("1. Prepare dataset")
        print("2. Train model")
        print("3. Evaluate model")
        print("4. Run full pipeline")
        print("5. Reconfigure setup")
        print("6. Cleanup generated files")
        print("0. Exit")
        choice = input("Select an option: ").strip()

        if choice == '1':
            print("Preparing dataset...")
            try:
                prepare_dataset(data_dir, save_numpy=True)
                print("Dataset preparation complete.")
            except ValueError as e:
                print(f"Dataset preparation failed: {e}")


        elif choice == '2':
            model_type = input("Select model type (cnn / svm / rf): ").strip().lower()

            if model_type == 'cnn':
                batch_size = int(input("Batch size [default: 32]: ") or 32)
                epochs = int(input("Number of epochs [default: 30]: ") or 30)
                patience = int(input("Early stopping patience [default: 5]: ") or 5)
                train_model(model_type, data_dir, model_path_cnn, batch_size, epochs, patience)

            elif model_type == 'svm':
                try:
                    run_feature_extraction(data_dir)
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    continue
                print("Training SVM model...")
                run_svm_training(model_path=model_path_svm)

            elif model_type == 'rf':
                try:
                    run_feature_extraction(data_dir)
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    continue
                print("Training Random Forest model...")
                run_rf_training(model_path=model_path_rf)

        elif choice == '3':
            model_type = input("Model type to evaluate (cnn / svm / rf / ensemble): ").strip().lower()
            output_dir_custom = input(f"Enter subdirectory name for this model [default: {model_type}]: ").strip()
            subfolder = output_dir_custom or model_type
            output_dir_eval = os.path.join(results_root, subfolder)

            if model_type == 'cnn':
                if not os.path.exists(model_path_cnn):
                    print(f"Model file '{model_path_cnn}' not found. Please train the model first.")
                    continue
                try:
                    (_, _), (_, _), (X_test, y_test) = prepare_dataset(data_dir, save_numpy=True)
                except ValueError as e:
                    print(f"Dataset preparation failed: {e}")
                    continue
                evaluate_model(model_path_cnn, X_test, y_test, output_dir=output_dir_eval)

            elif model_type == 'svm':
                if not os.path.exists(model_path_svm):
                    print(f"Model file '{model_path_svm}' not found. Please train the model first.")
                    continue
                run_svm_evaluation(model_path=model_path_svm, output_dir=output_dir_eval)

            elif model_type == 'rf':
                if not os.path.exists(model_path_rf):
                    print(f"Model file '{model_path_rf}' not found. Please train the model first.")
                    continue
                run_rf_evaluation(model_path=model_path_rf, output_dir=output_dir_eval)

            elif model_type == 'ensemble':
                run_ensemble_evaluation(results_root=results_root)

            else:
                print("Unsupported model type.")

        elif choice == '4':
            print("\n--- Full Pipeline ---")
            pipeline_choice = input("Select model(s) to run (cnn / svm / rf / all): ").strip().lower()

            valid_choices = {'cnn', 'svm', 'rf', 'all'}
            if pipeline_choice not in valid_choices:
                print("Invalid choice. Please select from: cnn / svm / rf / all.")
                continue

            interactive = input("Use interactive mode for parameters, i.e., require user input? (y/n): ").strip().lower() == 'y'

            # CNN
            if pipeline_choice in ('cnn', 'all'):
                print("\n--- Running CNN pipeline ---")
                try:
                    if interactive:
                        batch_size = int(input("Batch size [default: 32]: ") or 32)
                        epochs = int(input("Number of epochs [default: 30]: ") or 30)
                        patience = int(input("Early stopping patience [default: 5]: ") or 5)
                    else:
                        batch_size, epochs, patience = 32, 30, 5

                    (X_train, y_train), (X_val, y_val), (X_test, y_test) = prepare_dataset(data_dir, save_numpy=True)
                    train_model('cnn', data_dir, model_path_cnn, batch_size, epochs, patience)
                    evaluate_model(model_path_cnn, X_test, y_test, output_dir=os.path.join(results_root, "cnn"))
                except Exception as e:
                    print(f"CNN pipeline failed: {e}")

            # SVM / RF
            if pipeline_choice in ('svm', 'rf', 'all'):
                try:
                    run_feature_extraction(data_dir, force=not interactive)
                except Exception as e:
                    print(f"Feature extraction failed: {e}")
                    continue

            # SVM
            if pipeline_choice in ('svm', 'all'):
                print("\n--- Running SVM pipeline ---")
                try:
                    run_svm_training(
                        model_path=model_path_svm,
                        use_weighted_classes=None if interactive else False
                    )
                    run_svm_evaluation(
                        model_path=model_path_svm,
                        output_dir=os.path.join(results_root, "svm")
                    )
                except Exception as e:
                    print(f"SVM pipeline failed: {e}")

            # RF
            if pipeline_choice in ('rf', 'all'):
                print("\n--- Running Random Forest pipeline ---")
                try:
                    run_rf_training(
                        model_path=model_path_rf,
                        use_weighted_classes=None if interactive else True
                    )
                    run_rf_evaluation(
                        model_path=model_path_rf,
                        output_dir=os.path.join(results_root, "rf")
                    )
                except Exception as e:
                    print(f"Random Forest pipeline failed: {e}")

            print("\nSelected pipeline(s) complete.")

        elif choice == '5':
            print("\nReconfiguring setup...")
            data_dir, model_path_cnn, model_path_svm, model_path_rf, results_root = setup_paths()

        elif choice == '6':
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
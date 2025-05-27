import logging
import json
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from models import create_cnn_model
from preprocessing import prepare_dataset
from train_utils import load_cached_data  # NOVO

logging.basicConfig(level=logging.INFO)

def train_cnn(data_dir, model_save_path, batch_size=32, epochs=100, patience=10):
    """
    Treina um modelo CNN com os dados fornecidos.
    """
    logging.info("Preparando o dataset para treino da CNN...")

    cached = load_cached_data()
    if cached:
        (X_train, y_train), (X_val, y_val), _ = cached
    else:
        logging.info("[i] Dados não encontrados em cache. A processar imagens...")
        (X_train, y_train), (X_val, y_val), _ = prepare_dataset(data_dir, save_numpy=True)

    if len(X_train) == 0 or len(y_train) == 0:
        raise ValueError("Os dados de treinamento estão vazios.")
    if len(X_val) == 0 or len(y_val) == 0:
        raise ValueError("Os dados de validação estão vazios.")
    
    logging.info("Criando o modelo CNN...")
    model = create_cnn_model()
    
    callbacks = [
        ModelCheckpoint(model_save_path, save_best_only=True),
        EarlyStopping(patience=patience, restore_best_weights=True)
    ]
    
    datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    logging.info("Iniciando o treino...")
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks
    )
    
    logging.info("Treino concluído. A guardar o histórico...")
    with open('training_history.json', 'w') as f:
        json.dump(history.history, f)
    
    logging.info(f"[✓] Modelo CNN guardado em: {model_save_path}")
    return history

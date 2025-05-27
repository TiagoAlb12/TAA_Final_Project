from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

def create_cnn_model(input_shape=(224, 224, 1), num_classes=4, filters=[32, 64, 128], dropout_rates=[0.2, 0.3, 0.4], learning_rate=0.001):
    """
    Cria modelo CNN para classificação de Alzheimer com hiperparâmetros configuráveis.
    """
    model = Sequential()
    
    # Bloco 1
    model.add(Conv2D(filters[0], (3, 3), activation='relu', padding='same', input_shape=input_shape, kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters[0], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rates[0]))
    
    # Bloco 2
    model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters[1], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rates[1]))
    
    # Bloco 3
    model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Conv2D(filters[2], (3, 3), activation='relu', padding='same', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(dropout_rates[2]))
    
    # Classificação
    model.add(Flatten())
    model.add(Dense(256, activation='relu', kernel_regularizer=l2(0.01)))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    
    # Compilação
    model.compile(optimizer=Adam(learning_rate=learning_rate),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    return model
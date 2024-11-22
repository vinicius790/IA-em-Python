import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Carrega o conjunto de dados
def load_dataset():
    """
    Carrega o conjunto de dados Iris a partir de uma URL pública.

    Returns:
        pd.DataFrame: Conjunto de dados carregado.
    """
    url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
    dataset = pd.read_csv(url, names=attributes)
    return dataset

# Prepara os dados
def prepare_data(dataset):
    """
    Prepara os dados para treinamento, incluindo codificação de rótulos e divisão em conjuntos de treinamento e validação.

    Args:
        dataset (pd.DataFrame): O conjunto de dados original.

    Returns:
        tuple: Arrays numpy para características de treinamento, características de validação,
               rótulos de treinamento e rótulos de validação.
    """
    dataset['class'] = dataset['class'].astype('category').cat.codes
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    seed = 7
    X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size, random_state=seed)
    Y_train = to_categorical(Y_train)
    Y_validation = to_categorical(Y_validation)
    return X_train, X_validation, Y_train, Y_validation

# Define o modelo de rede neural
def create_nn_model():
    """
    Define a arquitetura da rede neural.

    Returns:
        Sequential: Modelo de rede neural compilado.
    """
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Treina o modelo de rede neural
def train_nn_model(X_train, Y_train):
    """
    Treina o modelo de rede neural usando os dados de treinamento.

    Args:
        X_train (np.array): Características de treinamento.
        Y_train (np.array): Rótulos de treinamento.

    Returns:
        Sequential: Modelo de rede neural treinado.
    """
    model = create_nn_model()
    
    # Adiciona callbacks para early stopping e salvamento de checkpoints do modelo
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    model_checkpoint = ModelCheckpoint('best_nn_model.h5', monitor='val_loss', save_best_only=True, verbose=1)
    
    history = model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_split=0.1, verbose=1, callbacks=[early_stopping, model_checkpoint])
    
    # Salva o modelo final
    model.save('neural_network_model.h5')
    print("Neural Network model saved as 'neural_network_model.h5'")
    return model, history

# Avalia o modelo de rede neural
def evaluate_nn_model(model, X_validation, Y_validation):
    """
    Avalia o modelo de rede neural usando os dados de validação.

    Args:
        model (Sequential): Modelo de rede neural treinado.
        X_validation (np.array): Características de validação.
        Y_validation (np.array): Rótulos de validação.
    """
    scores = model.evaluate(X_validation, Y_validation, verbose=0)
    print(f"\nNeural Network Accuracy: {scores[1]*100:.2f}%")

# Plota gráficos de treinamento e validação
def plot_training_history(history):
    """
    Plota a história de treinamento e validação do modelo.

    Args:
        history (History): História do treinamento retornada pelo método `fit` do Keras.
    """
    # Plot loss
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Treinamento')
    plt.plot(history.history['val_loss'], label='Validação')
    plt.title('Perda durante o Treinamento e Validação')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.legend()

    # Plot accuracy
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Treinamento')
    plt.plot(history.history['val_accuracy'], label='Validação')
    plt.title('Acurácia durante o Treinamento e Validação')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Função principal para rodar o fluxo de trabalho
def main():
    dataset = load_dataset()
    X_train, X_validation, Y_train, Y_validation = prepare_data(dataset)
    nn_model, history = train_nn_model(X_train, Y_train)
    evaluate_nn_model(nn_model, X_validation, Y_validation)
    plot_training_history(history)

if __name__ == "__main__":
    main()

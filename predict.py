import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical

# Função para carregar o dataset
def load_dataset(url, attributes):
    """
    Carrega o dataset a partir da URL fornecida e codifica as classes como inteiros.

    Args:
        url (str): URL do dataset.
        attributes (list): Lista de atributos/colunas do dataset.

    Returns:
        pd.DataFrame: Dataset carregado e modificado.
    """
    dataset = pd.read_csv(url, names=attributes)
    dataset['class'] = dataset['class'].astype('category').cat.codes
    return dataset

# Função para dividir o dataset
def split_dataset(dataset, test_size=0.20, random_state=7):
    """
    Divide o dataset em conjuntos de treinamento e validação.

    Args:
        dataset (pd.DataFrame): Dataset a ser dividido.
        test_size (float): Proporção do dataset a ser usado como validação.
        random_state (int): Semente para geração de números aleatórios.

    Returns:
        tuple: Arrays para características de treinamento, características de validação,
               rótulos de treinamento e rótulos de validação.
    """
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    return train_test_split(X, Y, test_size=test_size, random_state=random_state)

# Função para treinar o modelo XGBoost
def train_xgboost(X_train, Y_train):
    """
    Treina um modelo XGBoost com os dados de treinamento.

    Args:
        X_train (np.array): Características de treinamento.
        Y_train (np.array): Rótulos de treinamento.

    Returns:
        XGBClassifier: Modelo XGBoost treinado.
    """
    model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    model.fit(X_train, Y_train)
    return model

# Função para avaliar o modelo
def evaluate_model(model, X_validation, Y_validation):
    """
    Avalia o modelo usando os dados de validação.

    Args:
        model (classifier): Modelo a ser avaliado.
        X_validation (np.array): Características de validação.
        Y_validation (np.array): Rótulos de validação.
    """
    predictions = model.predict(X_validation)
    print("\nClassification Report:")
    print(classification_report(Y_validation, predictions))
    print("Confusion Matrix:")
    print(confusion_matrix(Y_validation, predictions))
    print(f"Accuracy Score: {accuracy_score(Y_validation, predictions):.3f}")

# Função para carregar e avaliar o modelo de rede neural
def evaluate_neural_network(model_path, X_validation, Y_validation):
    """
    Carrega e avalia o modelo de rede neural usando os dados de validação.

    Args:
        model_path (str): Caminho para o modelo de rede neural salvo.
        X_validation (np.array): Características de validação.
        Y_validation (np.array): Rótulos de validação.

    Returns:
        Sequential: Modelo de rede neural carregado.
    """
    model = load_model(model_path)
    scores = model.evaluate(X_validation, to_categorical(Y_validation), verbose=0)
    print(f"\nNeural Network Accuracy: {scores[1]*100:.2f}%")
    return model

# Função para fazer previsões com ambos os modelos
def make_predictions(sample, xgb_model, nn_model):
    """
    Faz previsões usando os modelos XGBoost e Rede Neural.

    Args:
        sample (list): Amostra para previsão.
        xgb_model (XGBClassifier): Modelo XGBoost treinado.
        nn_model (Sequential): Modelo de rede neural treinado.
    """
    xgb_prediction = xgb_model.predict([sample])
    print(f"XGBoost Prediction: {xgb_prediction[0]}")

    nn_prediction = nn_model.predict([sample])
    nn_prediction_class = nn_prediction.argmax(axis=-1)
    print(f"Neural Network Prediction: {nn_prediction_class[0]}")

# Parâmetros e URL do dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]

# Carregar e dividir o dataset
dataset = load_dataset(url, attributes)
X_train, X_validation, Y_train, Y_validation = split_dataset(dataset)

# Treinar e avaliar o modelo XGBoost
xgb_model = train_xgboost(X_train, Y_train)
evaluate_model(xgb_model, X_validation, Y_validation)

# Avaliar o modelo de rede neural
nn_model = evaluate_neural_network('neural_network_model.h5', X_validation, Y_validation)

# Exemplo de previsão
sample = [5.1, 3.5, 1.4, 0.2]
make_predictions(sample, xgb_model, nn_model)

import pandas as pd
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from typing import List

# Database setup
DATABASE_URL = "sqlite:///./iris.db"
Base = declarative_base()

class Iris(Base):
    __tablename__ = "iris"
    id = Column(Integer, primary_key=True, index=True)
    sepal_length = Column(Float, index=True)
    sepal_width = Column(Float, index=True)
    petal_length = Column(Float, index=True)
    petal_width = Column(Float, index=True)
    class_name = Column(String, index=True)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)

# FastAPI setup
app = FastAPI()

class IrisCreate(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    class_name: str

class IrisUpdate(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    class_name: str

# CRUD Operations
@app.post("/iris/", response_model=IrisCreate)
def create_iris(iris: IrisCreate):
    db = SessionLocal()
    db_iris = Iris(**iris.dict())
    db.add(db_iris)
    db.commit()
    db.refresh(db_iris)
    db.close()
    return db_iris

@app.get("/iris/", response_model=List[IrisCreate])
def read_iris():
    db = SessionLocal()
    iris = db.query(Iris).all()
    db.close()
    return iris

@app.put("/iris/{iris_id}", response_model=IrisCreate)
def update_iris(iris_id: int, iris: IrisUpdate):
    db = SessionLocal()
    db_iris = db.query(Iris).filter(Iris.id == iris_id).first()
    if db_iris is None:
        raise HTTPException(status_code=404, detail="Iris not found")
    for key, value in iris.dict().items():
        setattr(db_iris, key, value)
    db.commit()
    db.refresh(db_iris)
    db.close()
    return db_iris

@app.delete("/iris/{iris_id}", response_model=IrisCreate)
def delete_iris(iris_id: int):
    db = SessionLocal()
    db_iris = db.query(Iris).filter(Iris.id == iris_id).first()
    if db_iris is None:
        raise HTTPException(status_code=404, detail="Iris not found")
    db.delete(db_iris)
    db.commit()
    db.close()
    return db_iris

# Load the dataset
url = 'http://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
attributes = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
dataset = pd.read_csv(url, names=attributes)

# Data exploration
def explore_data(dataset):
    print(f"Dataset shape: {dataset.shape}")
    print("\nFirst 20 rows of the dataset:")
    print(dataset.head(20))
    print("\nDataset description:")
    print(dataset.describe())
    print("\nClass distribution:")
    print(dataset.groupby('class').size())

explore_data(dataset)

# Data visualization
def visualize_data(dataset):
    dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
    plt.suptitle('Box and Whisker Plots')
    plt.show()

    dataset.hist()
    plt.suptitle('Histograms')
    plt.show()

    scatter_matrix(dataset, figsize=(10, 10))
    plt.suptitle('Scatter Plot Matrix')
    plt.show()

visualize_data(dataset)

# Split dataset
def split_dataset(dataset):
    array = dataset.values
    X = array[:, 0:4]
    Y = array[:, 4]
    validation_size = 0.20
    seed = 7
    return train_test_split(X, Y, test_size=validation_size, random_state=seed)

X_train, X_validation, Y_train, Y_validation = split_dataset(dataset)

# Define models and pipelines
def define_models():
    models = {
        'Logistic Regression': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(solver='liblinear', multi_class='ovr'))
        ]),
        'Linear Discriminant Analysis': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', LinearDiscriminantAnalysis())
        ]),
        'K-Nearest Neighbors': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', KNeighborsClassifier())
        ]),
        'Decision Tree': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', DecisionTreeClassifier())
        ]),
        'Naive Bayes': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', GaussianNB())
        ]),
        'Support Vector Machine': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', SVC(gamma='auto'))
        ]),
        'XGBoost': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler()),
            ('clf', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'))
        ])
    }
    return models

models = define_models()

# Evaluate models
def evaluate_models(models, X_train, Y_train):
    results = []
    names = []
    seed = 7
    for name, model in models.items():
        kfold = KFold(n_splits=10, random_state=seed, shuffle=True)
        cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
        results.append(cv_results)
        names.append(name)
        print(f"{name}: {cv_results.mean():.3f} ({cv_results.std():.3f})")
    return results, names

results, names = evaluate_models(models, X_train, Y_train)

# Compare algorithms
def compare_algorithms(results, names):
    plt.figure(figsize=(12, 6))
    plt.boxplot(results, labels=names)
    plt.title('Algorithm Comparison')
    plt.ylabel('Accuracy')
    plt.show()

compare_algorithms(results, names)

# Neural Network
def create_nn_model():
    model = Sequential()
    model.add(Dense(12, input_dim=4, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def train_nn_model(X_train, Y_train):
    model = create_nn_model()
    history = model.fit(X_train, Y_train, epochs=150, batch_size=10, validation_split=0.1, verbose=1)
    model.save('neural_network_model.h5')
    print("Neural Network model saved as 'neural_network_model.h5'")
    return model, history

def evaluate_nn_model(model, X_validation, Y_validation):
    scores = model.evaluate(X_validation, Y_validation, verbose=0)
    print(f"\nNeural Network Accuracy: {scores[1]*100:.2f}%")

def plot_training_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss During Training and Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy During Training and Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

# Prepare data for neural network
Y_train_encoded = to_categorical(pd.factorize(Y_train)[0])
Y_validation_encoded = to_categorical(pd.factorize(Y_validation)[0])

nn_model, history = train_nn_model(X_train[_{{{CITATION{{{_1{](https://github.com/Ashuxyz/Python-for-Data-Science/tree/d6fdc63f5017f19b5bbecb1046d05f8230e3b481/ML%2Firis%2Firis.py)[_{{{CITATION{{{_2{](https://github.com/JANGHEEEUN/Keras/tree/6521457a36803a4d7252b408c33c6db992b92e2c/Keras%2FML%2Fm06_wine2_keras_answer.py)[_{{{CITATION{{{_3{](https://github.com/kindalime/Gerstein-Lab-Breakseq/tree/d9964d36fc929749e4a624db704248e3b9edad63/docker%2Fmodels%2Fscripts%2Fautoencoder.py)
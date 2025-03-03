# AVALIAÇÃO DO MODELO

from sklearn.metrics import classification_report
from training.training import train_model
import pandas as pd

"""  
x_test_vectorized: textos já vetorizados
y_test: Esses são os valores reais associados às amostras do conjunto de teste, que serão comparados com as previsões do modelo. EX: 0, 1, 1, 4, 5... equivalem ao rating de cada review

""" 
def evaluate_model(model, X_test_vectorized, y_test, class_names):
    y_pred = model.predict(X_test_vectorized)
    print("\nRelatório de classificação:")
    print(classification_report(y_test, y_pred, target_names=class_names))




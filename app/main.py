from pre_processing.pre_processor import load_df_processed
from training.training import train_model
from evaluation import evaluate_model
from visualization import plot_sentiment_distribution
from sklearn.metrics import classification_report
from training.training import train_model
from training.training_embedding import train_model

# Carregar dataset
df = load_df_processed()

# Treinar modelo
# model, vectorizer, X_test_vectorized, y_test = train_model(df)
model, vectorizer, X_test_vectorized, y_test = train_model(df)

# Avaliar modelo
y_pred = model.predict(X_test_vectorized)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=['péssimo', 'ruim', 'Neutro', 'bom', 'ótimo']))

# # Visualizar dados
# plot_sentiment_distribution(df)


# evaluate_model(model, X_test_vectorized, y_test, ['Negativo', 'Positivo', 'Neutro'])


from pre_processing import preprocess_portuguese_text
from dataset_example import load_example_dataset
from dataset_b2b import load_b2b_dataset
from dataset_0list import load_b2b_dataset
from training import train_model
from evaluation import evaluate_model
from visualization import plot_sentiment_distribution

# Carregar dataset
df = load_example_dataset()
df['processed_text'] = df['review_text'].apply(preprocess_portuguese_text)

# Treinar modelo
model, vectorizer, X_test_vectorized, y_test = train_model(df)

# Avaliar modelo
class_names = ['Negativo', 'Positivo', 'Neutro']
evaluate_model(model, X_test_vectorized, y_test, class_names)

# Visualizar dados
plot_sentiment_distribution(df)

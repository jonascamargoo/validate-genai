from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import numpy as np
import gensim.downloader as api
from gensim.models import Word2Vec
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier

# Função para gerar a média dos embeddings de palavras
def get_average_word2vec(tokens_list, model, vector_size):
    embeddings = []
    for tokens in tokens_list:
        word_vecs = [model[word] for word in tokens if word in model]
        if len(word_vecs) == 0:
            # Se nenhuma das palavras tem embedding, adiciona um vetor de zeros
            embeddings.append(np.zeros(vector_size))
        else:
            # Calcula a média dos vetores de palavras
            embeddings.append(np.mean(word_vecs, axis=0))
    return np.array(embeddings)

# PREPARAÇÃO PARA MODELAGEM UTILIZANDO EMBEDDINGS
def train_model(df):
    # Separa features (X) e target (y)
    X = df['processed_text'] # Features: coluna com os textos já pré-processados
    y = df['sentiment'] # Target: sentimentos (1, 2, 3, 4 e 5)

    # Divisão de treino e teste (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Baixa o modelo de Word2Vec (exemplo: Google News)
    word2vec_model = api.load("glove-wiki-gigaword-50")  # 50 dimensões, mas existem opções maiores.

    # Transforma os textos em embeddings
    X_train_vectorized = get_average_word2vec(X_train, word2vec_model, vector_size=50)
    X_test_vectorized = get_average_word2vec(X_test, word2vec_model, vector_size=50)

    # # Usa um classificador simples (ex: Naive Bayes)
    # model = MultinomialNB()

    # # Treina o modelo
    # model.fit(X_train_vectorized, y_train)
    
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train_vectorized, y_train)

    return model, word2vec_model, X_test_vectorized, y_test


    """Relatório de classificação:
        precision    recall  f1-score   support

        péssimo       0.48      0.60      0.53      5474
        ruim       0.80      0.02      0.04      1678
        Neutro       0.40      0.04      0.07      3263
        bom       0.34      0.19      0.25      6469
        ótimo       0.45      0.73      0.56      9591

        accuracy                           0.44     26475
        macro avg       0.49      0.32      0.29     26475
        weighted avg       0.45      0.44      0.38     26475
        
    """
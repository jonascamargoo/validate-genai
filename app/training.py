from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB

# PREPARAÇÃO PARA MODELAGEM UTILIZANDO BAG OF WORDS - talvez mudar para embeddings
def train_model(df):
    # Separa features (X) e target (y)
    X = df['processed_text'] # Features: coluna com os textos já pré-processados
    y = df['sentiment'] # Target: sentimentos (0, 1 ou 2) - coluna de rótulos de sentimento (exemplo: 0 = negativo, 1 = neutro, 2 = positivo).
    
    """
    X: Features (dados de entrada, como textos, imagens, etc.).
    
    y: Target (os rótulos ou classes dos dados, como "positivo", "negativo", "neutro").
    
    test_size=0.2: Define que 20% dos dados serão usados para teste, e os 80% restantes para treino.
    
    stratify=y: Mantém a proporção das classes nos conjuntos de treino e teste. Exemplo: se há 50% de textos positivos no dataset original, o conjunto de treino e teste terão essa mesma proporção.
    
    random_state=42: Define uma semente aleatória fixa para garantir que a divisão dos dados seja sempre a mesma em execuções diferentes.
    """  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    # Se tivermos 5.000 palavras no vocabulário, cada texto vira um vetor de tamanho 5.000, onde cada posição contém a contagem da respectiva palavra no texto.
    vectorizer = CountVectorizer(max_features=5000)
    X_train_vectorized = vectorizer.fit_transform(X_train)
    
    # Garante que se novas palavras surgirem no teste que não estavam no treino, elas serão ignoradas, evitando problemas no modelo
    X_test_vectorized = vectorizer.transform(X_test)

    model = MultinomialNB() # Naive Bayes

    """
    Aprende a relação entre palavras e classes: O modelo analisa quantas vezes cada palavra aparece em textos de cada classe.
    
    Calcula probabilidades: Ele estima a probabilidade de um texto pertencer a uma determinada classe com base na frequência das palavras.
    
    Cria um modelo treinado: Agora, o modelo pode prever a classe de um novo texto com base nas palavras presentes nele.
    """
    model.fit(X_train_vectorized, y_train)

    return model, vectorizer, X_test_vectorized, y_test

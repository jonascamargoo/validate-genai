import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer  # Stemmer para português
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer


# Baixar recursos do NLTK
nltk.download('stopwords')
nltk.download('punkt')

# Exemplo de dataset com avaliações em português
data = {
    'reviewText': [
        "Esse produto é incrível! Amei a qualidade e a entrega foi super rápida.",
        "O atendimento foi péssimo, nunca mais compro aqui.",
        "Produto razoável, poderia ser melhor, mas atende ao básico.",
        "Horrível! Chegou quebrado e ninguém resolveu meu problema.",
        "Ótima experiência, recomendo a todos!"
    ]
}

df = pd.DataFrame(data)

# PRE-PROCESSAMENTO
def preprocess_text(text):
    stop_words = set(stopwords.words('portuguese'))
    tokens = word_tokenize(text.lower())  # Converter para minúsculas e tokenizar
    filtered_tokens = [word for word in tokens if word.isalpha() and word not in stop_words]  # Remover stopwords e pontuação

    stemmer = RSLPStemmer()  # Stemmer para português
    stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

    return ' '.join(stemmed_tokens)

df['processedText'] = df['reviewText'].apply(preprocess_text)

# ANALISE DE SENTIMENTOS
analyzer = SentimentIntensityAnalyzer()

# Função para atribuir polaridade ao texto
def get_sentiment(text):
    scores = analyzer.polarity_scores(text)
    if scores['compound'] >= 0.05:
        return 'Positivo'
    elif scores['compound'] <= -0.05:
        return 'Negativo'
    else:
        return 'Neutro'

df['sentiment'] = df['processedText'].apply(get_sentiment)


print(df[['reviewText', 'sentiment']])
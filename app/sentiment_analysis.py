import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer, WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('punkt_tab')
# nltk.download('rslp')

# dataset
# data = {
#     'reviewText': [
#         "Esse produto é incrível! Amei a qualidade e a entrega foi super rápida.",
#         "O atendimento foi péssimo, nunca mais compro aqui.",
#         "Produto razoável, poderia ser melhor, mas atende ao básico.",
#         "Ótima experiência, recomendo a todos!",
#         "Horrível! Chegou quebrado e ninguém resolveu meu problema."
#     ]
# }

dataset = pd.read_csv('https://raw.githubusercontent.com/pycaret/pycaret/master/datasets/amazon.csv')
df = pd.DataFrame(dataset)

# PRE-PROCESSAMENTO
def preprocess_text(text):
    
    # Converter para minúsculas e tokenizar
    tokens = word_tokenize(text.lower())  

    # Selecionando as stopwords em portugues
    stop_words = set(stopwords.words('portuguese'))

    # Remover stopwords e pontuação
    # filtered_tokens = [token for token in tokens if word.isalpha() and word not in stop_words] o isalpha estava prejudicando a semantica...
    filtered_tokens = [token for token in tokens if token not in stop_words]

    # Stemmer para português
    # stemmer = RSLPStemmer()
    # stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens] Reduz as palavras ao seu radical (stem), muitas vezes removendo sufixos sem considerar a gramática da palavra. Isso prejudica a semântica... vou optar por lemetizar

    # Lemmatize the tokens
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    processed_text = ' '.join(lemmatized_tokens)

    return processed_text

df['processedText'] = df['reviewText'].apply(preprocess_text)

# # ANALISE DE SENTIMENTOS
# analyzer = SentimentIntensityAnalyzer()

# # Função para atribuir polaridade ao texto
# def get_sentiment(text):
#     scores = analyzer.polarity_scores(text)
#     if scores['compound'] >= 0.05:
#         return 'Positivo'
#     elif scores['compound'] <= -0.05:
#         return 'Negativo'
#     else:
#         return 'Neutro'

# df['sentiment'] = df['processedText'].apply(get_sentiment)

# print(df[['reviewText', 'sentiment']])
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

def preprocess_portuguese_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\d+', '', text)

        tokens = word_tokenize(text, language='portuguese')
        stop_words = set(stopwords.words('portuguese'))
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]

        stemmer = RSLPStemmer()
        return ' '.join(stemmer.stem(token) for token in filtered_tokens)
    return ""

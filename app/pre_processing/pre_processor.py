import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import RSLPStemmer

def remove_urls(text):
    # Substitui qualquer URL (começando com http:// ou https:// ou www.) por string vazia
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def remove_special_characters(text):
    # Remove pontuações e caracteres especiais, mantendo apenas letras e espaços
    # Pode ser prejudicial para a semântica
    return re.sub(r'[^\w\sáéíóúãõçÁÉÍÓÚÃÕÇ]', '', text)

def remove_numbers(text):
    # Remover números - substitui sequências de dígitos por string vazia
    return re.sub(r'\d+', '', text)

def tokenize_text(text):
    # Tokenizar - dividir o texto em palavras individuais usando tokenizador específico para português
    return word_tokenize(text, language='portuguese')

def remove_stopwords_and_short_tokens(tokens):
    # Remove stopwords e palavras muito curtas (menos de 3 caracteres)
    stop_words = set(stopwords.words('portuguese'))
    filtered_tokens = []
    for token in tokens:
        # Verifica se a palavra não é uma stopword e tem mais de 2 caracteres
        if token not in stop_words and len(token) > 2:
            filtered_tokens.append(token)
    
    return filtered_tokens

def apply_stemming(tokens):
    """
    Aplica stemming para reduzir palavras ao seu radical.
    
    Aplicar stemming - reduzir palavras ao seu radical. Ex: "comprei", "comprando", "compra" -> "compr". Isso ajuda a unificar variações de uma mesma palavra
    
    O stemmer pode prejudicar a legibilidade do texto, mas ajuda a melhorar a eficácia do modelo. Devo verificar com e sem o stemming, já que pretendo validar textos curtos
    
    """
    stemmer = RSLPStemmer()
    return [stemmer.stem(token) for token in tokens]

def pre_process_portuguese(text):
    # Função principal que realiza o pré-processamento completo do texto.
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = remove_urls(text)
    text = remove_special_characters(text)
    text = remove_numbers(text)
    tokens = tokenize_text(text)
    tokens = remove_stopwords_and_short_tokens(tokens)
    tokens = apply_stemming(tokens) # Juntar as palavras processadas de volta em um texto único separado por espaços
    return ' '.join(tokens)


# dataset real de avaliações de produtos da B2W (Americanas, Submarino, Shoptime). Link para download: https://github.com/americanas-tech/b2w-reviews01
import pandas as pd
import os


def load_df_processed():
    """
    Carrega o dataset de avaliações da B2W e converte as notas em categorias de sentimento.
    
    Retorna:
    - df (DataFrame): DataFrame processado com as colunas ['overall_rating', 'review_title', 'sentiment'].
    """
    # 🔹 Obtém o diretório onde este script está salvo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 🔹 Caminho absoluto do dataset
    dataset_path = os.path.join(script_dir, "B2W-Reviews01.csv")
    
    try:
        # Carrega o dataset
        df = pd.read_csv(dataset_path, encoding="utf-8")
        
        # Filtra colunas relevantes
        df_subset = df[["overall_rating", "review_text"]].copy()
        df_subset.rename(columns={"overall_rating": "sentiment"}, inplace=True)
        
        # Aplica o pré-processamento na coluna 'review_text' e substitui o conteúdo original
        df_subset['processed_text'] = df_subset['review_text'].apply(pre_process_portuguese)
        
        # Remove a coluna 'review_text' e mantém apenas o texto processado
        df_subset.drop(columns=['review_text'], inplace=True)
        
        return df_subset

    except FileNotFoundError:
        print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
        return None
    
    
# Exibe o dataset processado
# df_processed = load_df_processed()
# print(df_processed.head())
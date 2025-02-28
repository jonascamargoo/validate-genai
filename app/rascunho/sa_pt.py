# Importação das bibliotecas necessárias
import pandas as pd  # Pandas para manipulação de dados tabulares
import nltk  # Natural Language Toolkit - biblioteca principal para processamento de linguagem natural
import re  # Módulo de expressões regulares para manipulação de strings
import matplotlib.pyplot as plt  # Biblioteca para criar visualizações e gráficos
import seaborn as sns  # Biblioteca de visualização estatística baseada em matplotlib
from nltk.corpus import stopwords  # Coleção de palavras comuns que geralmente são removidas em PLN
from nltk.tokenize import word_tokenize  # Função para dividir texto em palavras individuais
from nltk.stem import RSLPStemmer  # Stemmer específico para português (reduz palavras ao radical)
from sklearn.feature_extraction.text import CountVectorizer  # Converte texto em vetores de contagem de palavras
from sklearn.model_selection import train_test_split  # Função para dividir dados em conjuntos de treino e teste
from sklearn.naive_bayes import MultinomialNB  # Algoritmo Naive Bayes para classificação de texto
from sklearn.metrics import classification_report, confusion_matrix  # Métricas de avaliação para classificação
import numpy as np  # Biblioteca para computação numérica em Python

# Bloco para download de recursos do NLTK necessários para processamento em português
# Estes downloads são necessários apenas na primeira execução
# Eles baixam recursos como listas de stopwords, modelos de tokenização e o stemmer para português
# nltk.download('stopwords')  # Baixa listas de palavras comuns em vários idiomas, incluindo português
# nltk.download('punkt')  # Baixa modelo de tokenização treinado para segmentar texto em frases e palavras
# nltk.download('rslp')  # Baixa o stemmer RSLP para português brasileiro

# Definição da função de pré-processamento de texto em português
def preprocess_portuguese_text(text):
    """
    Função que realiza o pré-processamento completo de texto em português para análise de sentimentos.
    
    Parâmetros:
    text (str): O texto a ser processado
    
    Retorna:
    str: Texto processado, com palavras normalizadas e filtradas
    """
    # Verificamos se o texto é uma string válida (evita erros com NaN ou valores numéricos)
    if isinstance(text, str):
        # Converter todo o texto para minúsculas para padronização
        text = text.lower()
        
        # Remover URLs usando expressão regular
        # Substitui qualquer URL (começando com http:// ou https:// ou www.) por string vazia
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
        
        # Remover pontuações e símbolos especiais - mantém apenas letras e espaços
        text = re.sub(r'[^\w\s]', '', text)
        
        # Remover números - substitui sequências de dígitos por string vazia
        text = re.sub(r'\d+', '', text)
        
        # Tokenizar - dividir o texto em palavras individuais usando tokenizador específico para português
        tokens = word_tokenize(text, language='portuguese')
        
        # Remover stopwords (palavras comuns como "e", "de", "para" que não agregam significado semântico)
        # E também remover palavras muito curtas (menos de 3 caracteres)
        stop_words = set(stopwords.words('portuguese'))
        filtered_tokens = [token for token in tokens if token not in stop_words and len(token) > 2]
        
        # Aplicar stemming - reduzir palavras ao seu radical
        # Ex: "comprei", "comprando", "compra" -> "compr"
        # Isso ajuda a unificar variações de uma mesma palavra
        stemmer = RSLPStemmer()  # Stemmer específico para português brasileiro
        stemmed_tokens = [stemmer.stem(token) for token in filtered_tokens]
        
        # Juntar as palavras processadas de volta em um texto único separado por espaços
        processed_text = ' '.join(stemmed_tokens)
        return processed_text
    else:
        # Se o texto não for uma string válida, retornamos uma string vazia
        return ""

# SEÇÃO DE DADOS - Três opções de datasets são fornecidas, escolha uma delas

# Opção 1: Dataset de exemplo pequeno para testes rápidos
# Este é um pequeno conjunto de dados com 5 exemplos já rotulados para testar o pipeline
data_example = {
    'review_text': [
        "Esse produto é incrível! Amei a qualidade e a entrega foi super rápida.",  # Review positiva
        "O atendimento foi péssimo, nunca mais compro aqui.",  # Review negativa
        "Produto razoável, poderia ser melhor, mas atende ao básico.",  # Review neutra
        "Ótima experiência, recomendo a todos!",  # Review positiva
        "Horrível! Chegou quebrado e ninguém resolveu meu problema."  # Review negativa
    ],
    'sentiment': [1, 0, 2, 1, 0]  # Códigos: 1 = Positivo, 0 = Negativo, 2 = Neutro
}
# Convertemos o dicionário em um DataFrame do pandas
df_example = pd.DataFrame(data_example)

# Opção 2: Dataset b2w-reviews (descomente as linhas abaixo para usar)
# Este é um dataset real de avaliações de produtos da B2W (Americanas, Submarino, Shoptime)
# Link para download: https://github.com/americanas-tech/b2w-reviews01
# df = pd.read_csv('b2w-reviews01.csv', encoding='utf-8')  # Carrega o arquivo CSV
# # Transforma as notas (1-5) em sentimentos: 
# # - Notas > 3 são positivas (1)
# # - Notas < 3 são negativas (0)
# # - Nota 3 é considerada neutra (2)
# df['sentiment'] = df['rating'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))
# df = df.rename(columns={'text': 'review_text'})  # Renomeia a coluna para padronizar

# Opção 3: Dataset Olist (descomente as linhas abaixo para usar)
# Dataset de e-commerce brasileiro com reviews de clientes
# Link: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce
# df = pd.read_csv('olist_order_reviews_dataset.csv', encoding='utf-8')  # Carrega o arquivo CSV
# df = df.rename(columns={'review_comment_message': 'review_text'})  # Renomeia a coluna para padronizar
# # Mapeia o score para as categorias de sentimento
# df['sentiment'] = df['review_score'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))
# df = df.dropna(subset=['review_text'])  # Remove linhas onde o texto da review está vazio

# Aqui escolhemos qual dataset vamos usar - por padrão, usamos o dataset de exemplo
df = df_example.copy()  # Fazemos uma cópia para não modificar o original

# Aplicamos o pré-processamento em todos os textos do dataset
# Criamos uma nova coluna 'processed_text' com os textos já pré-processados
df['processed_text'] = df['review_text'].apply(preprocess_portuguese_text)

# Mostra exemplos de textos originais e processados para verificar o pré-processamento
print("Exemplos de textos processados:")
for i, row in df.head().iterrows():  # Percorre as primeiras linhas do DataFrame
    print(f"Original: {row['review_text']}")  # Texto original
    print(f"Processado: {row['processed_text']}")  # Texto após pré-processamento
    # Traduz o código numérico do sentimento para texto
    print(f"Sentimento: {'Positivo' if row['sentiment'] == 1 else ('Negativo' if row['sentiment'] == 0 else 'Neutro')}")
    print("-" * 50)  # Linha separadora para melhor visualização

# PREPARAÇÃO PARA MODELAGEM

# Separa features (X) e target (y)
X = df['processed_text']  # Features: textos pré-processados
y = df['sentiment']  # Target: sentimentos (0, 1 ou 2)

# Verifica quais classes estão presentes nos dados
# Isso é importante porque nem sempre temos exemplos de todas as classes
present_classes = sorted(df['sentiment'].unique())  # Obtém valores únicos e os ordena
print(f"\nClasses presentes nos dados: {present_classes}")

# Cria dinamicamente os nomes das classes baseado nas classes presentes nos dados
# Isso evita o erro de incompatibilidade entre classes presentes e nomes fornecidos
class_names = []
for cls in present_classes:
    if cls == 0:
        class_names.append('Negativo')
    elif cls == 1:
        class_names.append('Positivo')
    elif cls == 2:
        class_names.append('Neutro')

print(f"Nomes das classes para relatório: {class_names}")

# Divisão em conjuntos de treino e teste
# Usamos stratify=y para garantir que a proporção de classes seja mantida nas divisões
# Random_state garante reprodutibilidade (sempre teremos a mesma divisão)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verificamos se todas as classes estão representadas em ambos os conjuntos
print(f"Classes no conjunto de treino: {sorted(y_train.unique())}")
print(f"Classes no conjunto de teste: {sorted(y_test.unique())}")

# VETORIZAÇÃO - Transformação de texto em representação numérica

# CountVectorizer transforma textos em uma matriz esparsa de contagem de palavras
# max_features=5000 limita o vocabulário às 5000 palavras mais frequentes
vectorizer = CountVectorizer(max_features=5000)

# Aprende o vocabulário a partir dos dados de treino e transforma esses dados
X_train_vectorized = vectorizer.fit_transform(X_train)

# Transforma os dados de teste usando o vocabulário aprendido nos dados de treino
X_test_vectorized = vectorizer.transform(X_test)

# TREINAMENTO DO MODELO

# Usamos Naive Bayes Multinomial - um classificador eficiente para texto
model = MultinomialNB()  # Instancia o modelo
model.fit(X_train_vectorized, y_train)  # Treina o modelo com os dados de treino

# AVALIAÇÃO DO MODELO

# Predição no conjunto de teste
y_pred = model.predict(X_test_vectorized)

# Imprime relatório de classificação detalhado (precision, recall, f1-score)
print("\nRelatório de classificação:")
print(classification_report(y_test, y_pred, target_names=class_names))

# FUNÇÃO PARA ANALISAR NOVOS TEXTOS

def analyze_sentiment(text):
    """
    Analisa o sentimento de um novo texto usando o modelo treinado.
    
    Parâmetros:
    text (str): O texto a ser analisado
    
    Retorna:
    str: O sentimento predito ('Positivo', 'Negativo' ou 'Neutro')
    """
    # Pré-processa o texto da mesma forma que os dados de treino
    processed = preprocess_portuguese_text(text)
    
    # Vetoriza o texto processado usando o mesmo vocabulário
    vectorized = vectorizer.transform([processed])
    
    # Faz a predição usando o modelo treinado
    prediction = model.predict(vectorized)[0]
    
    # Mapeia o código numérico para texto
    sentiment_map = {0: 'Negativo', 1: 'Positivo', 2: 'Neutro'}
    return sentiment_map[prediction]

# Testa a função com alguns exemplos
test_texts = [
    "O produto superou minhas expectativas, recomendo!",  # Exemplo positivo
    "Decepcionante, não funciona como prometido.",  # Exemplo negativo
    "Produto ok, nada demais."  # Exemplo neutro
]

# Analisa cada texto de teste e imprime o resultado
print("\nAnálise de novos textos:")
for text in test_texts:
    sentiment = analyze_sentiment(text)
    print(f"Texto: {text}")
    print(f"Sentimento: {sentiment}")
    print("-" * 30)

# VISUALIZAÇÕES

# Gráfico de barras mostrando a distribuição de sentimentos no dataset
plt.figure(figsize=(8, 5))  # Define o tamanho da figura
# Converte os códigos numéricos em nomes legíveis e conta cada categoria
sentiment_counts = df['sentiment'].map({0: 'Negativo', 1: 'Positivo', 2: 'Neutro'}).value_counts()
# Cria o gráfico de barras
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values)
plt.title('Distribuição de Sentimentos')  # Título do gráfico
plt.ylabel('Contagem')  # Rótulo do eixo y
plt.xlabel('Sentimento')  # Rótulo do eixo x
plt.tight_layout()  # Ajusta o layout
plt.savefig('sentiment_distribution.png')  # Salva a figura em um arquivo
plt.close()  # Fecha a figura para liberar memória

# Função para visualizar as palavras mais comuns por sentimento
def plot_top_words(df, sentiment_value, sentiment_name, top_n=15):
    """
    Cria um gráfico das palavras mais frequentes para um determinado sentimento.
    
    Parâmetros:
    df (DataFrame): O DataFrame com os dados
    sentiment_value (int): O valor numérico do sentimento (0, 1 ou 2)
    sentiment_name (str): O nome do sentimento ('Positivo', 'Negativo', 'Neutro')
    top_n (int): Número de palavras a serem mostradas
    """
    # Verificar se há exemplos desta classe no dataset
    if sentiment_value not in df['sentiment'].values:
        print(f"Não há exemplos da classe {sentiment_name} para visualizar")
        return
    
    # Filtrar textos por sentimento e juntá-los em uma string única
    sentiment_texts = ' '.join(df[df['sentiment'] == sentiment_value]['processed_text'])
    
    # Verificar se há texto para analisar
    if not sentiment_texts.strip():
        print(f"Sem palavras para analisar na classe {sentiment_name}")
        return
    
    # Dividir o texto em palavras individuais
    words = sentiment_texts.split()
    if not words:
        print(f"Sem palavras para analisar na classe {sentiment_name}")
        return
    
    # Contar a frequência de cada palavra e selecionar as mais comuns
    word_counts = pd.Series(words).value_counts().head(top_n)
    
    # Criar o gráfico de barras
    plt.figure(figsize=(10, 6))
    sns.barplot(x=word_counts.values, y=word_counts.index)
    plt.title(f'Top {top_n} palavras em reviews {sentiment_name}')
    plt.xlabel('Contagem')
    plt.tight_layout()
    plt.savefig(f'top_words_{sentiment_name.lower()}.png')
    plt.close()

# Gerar visualizações para cada classe presente nos dados
for class_val, class_name in zip(present_classes, class_names):
    plot_top_words(df, class_val, class_name)

print("\nAnálise concluída! Visualizações salvas em arquivos .png")
# Opção 3: Dataset Olist (descomente as linhas abaixo para usar)
# Dataset de e-commerce brasileiro com reviews de clientes
# Link: https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce


import pandas as pd

df = pd.read_csv('olist_order_reviews_dataset.csv', encoding='utf-8')  # Carrega o arquivo CSV
df = df.rename(columns={'review_comment_message': 'review_text'})  # Renomeia a coluna para padronizar

# Mapeia o score para as categorias de sentimento
df['sentiment'] = df['review_score'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))
df = df.dropna(subset=['review_text'])  # Remove linhas onde o texto da review está vazio
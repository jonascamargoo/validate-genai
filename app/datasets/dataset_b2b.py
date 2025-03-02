# dataset real de avaliações de produtos da B2W (Americanas, Submarino, Shoptime). Link para download: https://github.com/americanas-tech/b2w-reviews01

import pandas as pd

# Carrega o dataset de reviews
df = pd.read_csv("B2W-Reviews01.csv", encoding="utf-8")

# Converter a nota (rating) em categorias de sentimento:
#   - Positivo (1) para ratings acima de 3
#   - Neutro (2) para ratings igual a 3
#   - Negativo (0) para ratings abaixo de 3
df['sentiment'] = df['overall_rating'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))
# print(df.head())

df_subset = df[['overall_rating', 'review_title', 'review_text']]
print(df_subset)
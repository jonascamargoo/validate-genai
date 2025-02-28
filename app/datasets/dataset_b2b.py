# import pandas as pd

# # Opção 2: Dataset b2w-reviews
# # Este é um dataset real de avaliações de produtos da B2W (Americanas, Submarino, Shoptime)
# # Link para download: https://github.com/americanas-tech/b2w-reviews01

# # Transforma as notas (1-5) em sentimentos: 
# # - Notas > 3 são positivas (1)
# # - Notas < 3 são negativas (0)
# # - Nota 3 é considerada neutra (2)


# df = pd.read_csv('b2w-reviews01.csv', encoding='utf-8')  # Carrega o arquivo CSV
# df['sentiment'] = df['rating'].apply(lambda x: 1 if x > 3 else (0 if x < 3 else 2))
# df = df.rename(columns={'text': 'review_text'})  # Renomeia a coluna para padronizar

    
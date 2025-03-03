# dataset real de avaliações de produtos da B2W (Americanas, Submarino, Shoptime). Link para download: https://github.com/americanas-tech/b2w-reviews01
import pandas as pd
import os

def load_b2w_dataframe():
    """
    Carrega o dataset de avaliações da B2W e converte as notas em categorias de sentimento.
    
    Retorna:
    - df (DataFrame): DataFrame processado com as colunas ['overall_rating', 'review_title', 'review_text', 'sentiment'].
    """
    # 🔹 Obtém o diretório onde este script está salvo
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 🔹 Caminho absoluto do dataset
    dataset_path = os.path.join(script_dir, "B2W-Reviews01.csv")
    
    try:
        # Carrega o dataset
        df = pd.read_csv(dataset_path, encoding="utf-8")
        
        # Filtra colunas relevantes
        df_subset = df[["overall_rating", "review_title", "review_text"]]
        
        return df_subset

    except FileNotFoundError:
        print(f"Erro: O arquivo {dataset_path} não foi encontrado.")
        return None
    
    
# Exibe o dataset
print(load_b2w_dataframe())
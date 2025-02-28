import pandas as pd

# Opção 1: Dataset de exemplo pequeno para testes rápidos
# Este é um pequeno conjunto de dados com 5 exemplos já rotulados para testar o pipeline
def load_example_dataset():
    data = {
        'review_text': [
            "Esse produto é incrível!",
            "O atendimento foi péssimo.",
            "Produto razoável.",
            "Ótima experiência!",
            "Horrível! Chegou quebrado."
        ],
        'sentiment': [1, 0, 2, 1, 0]
    }
    return pd.DataFrame(data)

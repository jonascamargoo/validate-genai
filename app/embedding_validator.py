import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from pre_processing.pre_processor import pre_process_portuguese

class EmbeddingValidator:
    def __init__(self, modelo='all-MiniLM-L6-v2'):
        """
        Inicializa o validador com modelo de embeddings.
        
        Args:
            modelo (str): Modelo de embedding a ser usado
        """
        self.modelo = SentenceTransformer(modelo)
        
        # Respostas de referência para comparação
        self.respostas_referencia = [
            "Você tem um exame de ULTRASSONOGRAFIA realizado em 31/01/2025. O laudo está previsto para 03/02/2025.",
            "Encontrei um exame de ULTRASSONOGRAFIA realizado em 31/01/2025. Seu laudo está previsto para 03/02/2025.",
            "Verifiquei um ultrassom marcado para 31/01/2025, com laudo previsto em 03/02/2025."
        ]
        
        # Aplicar pré-processamento
        self.respostas_referencia = [
            pre_process_portuguese(resposta) for resposta in self.respostas_referencia
        ]
        
        # Pré-computar embeddings das respostas de referência
        self.embeddings_referencia = self.modelo.encode(self.respostas_referencia)
    
    def calcular_similaridade(self, resposta_chatbot):
        """
        Calcula a similaridade da resposta com padrões de referência.
        
        Args:
            resposta_chatbot (str): Resposta do chatbot a ser validada
        
        Returns:
            dict: Dicionário com métricas de validação
        """
        
        
        # # Pré-processar resposta do chatbot
        # resposta_processada = pre_process_portuguese(resposta_chatbot)
        # # Gerar embedding da resposta do chatbot
        
        
        embedding_resposta = self.modelo.encode([resposta_chatbot])
        
        # Calcular similaridade de cosseno. Calcula o "ângulo" entre os vetores.Quanto mais próximo de 1, mais similar
        similaridades = cosine_similarity(embedding_resposta, self.embeddings_referencia)[0]
        
        
        # Duvidoso!
        nota_max = max(similaridades)
        nota = round(nota_max * 5, 2)
        
        return {
            "nota": nota,
            "similaridades": list(similaridades),
            "resposta_mais_similar": self.respostas_referencia[np.argmax(similaridades)],
            "detalhes_validacao": {
                "tipo_exame_encontrado": "ULTRASSONOGRAFIA" in resposta_chatbot,
                "data_realizacao_encontrada": "31/01/2025" in resposta_chatbot,
                "data_laudo_encontrada": "03/02/2025" in resposta_chatbot
            }
        }


# Exemplo de uso
def demonstrar_validacao():
    validator = EmbeddingValidator()
    
    exemplos_resposta = [
        "Você tem um exame de ULTRASSONOGRAFIA realizado em 31/01/2025. O laudo está previsto para 03/02/2025.",
        "Olá! Encontrei seu exame de imagem para o próximo mês.",
        "Bom dia! Seu ultrassom foi agendado e será processado em breve."
    ]
    
    for resposta_chatbot in exemplos_resposta:
        resultado = validator.calcular_similaridade(resposta_chatbot)
        print(f"\nResposta: {resposta_chatbot}")
        print(f"Nota de Validação: {resultado['nota']}/5")
        print(f"Resposta Mais Similar: {resultado['resposta_mais_similar']}")
        print("Detalhes de Validação:", resultado['detalhes_validacao'])

if __name__ == "__main__":
    demonstrar_validacao()
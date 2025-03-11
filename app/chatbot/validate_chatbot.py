import re
import json
import requests
from typing import List, Dict, Any, Tuple

class ChatbotIAValidator:
    def __init__(self, api_key=None, endpoint=None):
        """
        Inicializa o validador usando IA para comparar respostas.
        
        Args:
            api_key: Chave da API (OpenAI, Azure, Claude, etc.)
            endpoint: Endpoint da API (opcional, caso não use OpenAI padrão)
        """
        self.api_key = api_key
        self.endpoint = endpoint
        
        # Informações esperadas e restrições para validação
        self.informacoes_esperadas = {
            "tipo_exame": "ULTRASSONOGRAFIA",
            "data_realizacao": "31/01/2025",
            "data_laudo": "03/02/2025",
        }
        
        # Prompt para direcionar o modelo de IA
        self.template_prompt = """
        Você é um validador de respostas de chatbot especializado em informações médicas. 
        Analise a resposta abaixo e verifique se ela contém todas as informações essenciais 
        sobre o exame médico, independentemente de variações na formulação.
        
        Resposta a analisar: 
        "{resposta}"
        
        Informações que devem estar presentes:
        - Tipo de exame: {tipo_exame}
        - Data de realização: {data_realizacao}
        - Data prevista para o laudo: {data_laudo}
        
        Retorne APENAS um objeto JSON com o seguinte formato:
        {{
            "tipo_exame_presente": bool,
            "data_realizacao_presente": bool,
            "data_laudo_presente": bool,
            "informacoes_corretas": bool,
            "confianca": float (entre 0 e 1),
            "observacoes": string
        }}
        """
    
    def extrair_mensagens_chatbot(self, texto_conversa: str) -> List[str]:
        """Extrai apenas as mensagens do chatbot de uma conversa do WhatsApp."""
        mensagens = []
        linhas = texto_conversa.strip().split('\n')
        
        for linha in linhas:
            if "Futurotec Homologação:" in linha:
                # Extrai a mensagem após o prefixo
                prefixo, _, mensagem = linha.partition(': ')
                if mensagem.strip():
                    mensagens.append(mensagem.strip())
        
        return mensagens
    
    def consultar_ia(self, resposta: str) -> Dict[str, Any]:
        """
        Consulta modelo de IA para validar a resposta.
        
        Este exemplo usa a API da OpenAI, mas pode ser adaptado para outras.
        """
        if not self.api_key:
            # Simulação da resposta da IA para testes sem API
            # Em produção, substitua pela chamada real à API
            return self._simular_resposta_ia(resposta)
        
        prompt = self.template_prompt.format(
            resposta=resposta,
            **self.informacoes_esperadas
        )
        
        # Estrutura para OpenAI API - adapte para outras APIs conforme necessário
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        data = {
            "model": "gpt-4",  # ou outro modelo adequado
            "messages": [
                {"role": "system", "content": "Você é um assistente especializado em validação de dados médicos."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,  # baixa temperatura para respostas mais consistentes
            "response_format": {"type": "json_object"}  # solicita resposta em JSON
        }
        
        try:
            endpoint = self.endpoint or "https://api.openai.com/v1/chat/completions"
            response = requests.post(endpoint, headers=headers, json=data)
            response.raise_for_status()
            result = response.json()
            
            # Extrair o conteúdo da resposta - adapte conforme a API
            content = result["choices"][0]["message"]["content"]
            return json.loads(content)
        except Exception as e:
            print(f"Erro ao consultar a API: {e}")
            return {
                "tipo_exame_presente": False,
                "data_realizacao_presente": False,
                "data_laudo_presente": False,
                "informacoes_corretas": False,
                "confianca": 0.0,
                "observacoes": f"Erro na consulta à API: {str(e)}"
            }
    
    def _simular_resposta_ia(self, resposta: str) -> Dict[str, Any]:
        """
        Simula a resposta da IA para testes sem API.
        Em produção, esta função é substituída pela chamada real à API.
        """
        # Verificações simples para simular a análise da IA
        tipo_exame_presente = self.informacoes_esperadas["tipo_exame"] in resposta
        data_realizacao_presente = self.informacoes_esperadas["data_realizacao"] in resposta
        data_laudo_presente = self.informacoes_esperadas["data_laudo"] in resposta
        
        informacoes_corretas = tipo_exame_presente and data_realizacao_presente and data_laudo_presente
        
        # Cálculo de confiança simulado
        confianca = 0.0
        if informacoes_corretas:
            confianca = 0.95
        elif (tipo_exame_presente and data_realizacao_presente) or (tipo_exame_presente and data_laudo_presente):
            confianca = 0.7
        elif tipo_exame_presente or data_realizacao_presente or data_laudo_presente:
            confianca = 0.4
        
        observacoes = "Validação simulada (sem API). "
        if informacoes_corretas:
            observacoes += "Todas as informações essenciais estão presentes."
        else:
            faltantes = []
            if not tipo_exame_presente:
                faltantes.append("tipo de exame")
            if not data_realizacao_presente:
                faltantes.append("data de realização")
            if not data_laudo_presente:
                faltantes.append("data prevista do laudo")
            observacoes += f"Informações faltantes: {', '.join(faltantes)}."
        
        return {
            "tipo_exame_presente": tipo_exame_presente,
            "data_realizacao_presente": data_realizacao_presente,
            "data_laudo_presente": data_laudo_presente,
            "informacoes_corretas": informacoes_corretas,
            "confianca": confianca,
            "observacoes": observacoes
        }
    
    def validar_resposta(self, resposta: str) -> Dict[str, Any]:
        """Valida uma única resposta usando IA."""
        # Primeiro faz uma validação rápida para ver se a mensagem contém "exame"
        # Isso economiza chamadas à API para mensagens irrelevantes
        if "exame" not in resposta.lower():
            return {
                "mensagem_original": resposta,
                "relevante": False,
                "validacao_ia": None,
                "resultado_geral": False,
                "observacoes": "Mensagem não relacionada a exames."
            }
        
        # Consulta a IA para validar a resposta
        validacao_ia = self.consultar_ia(resposta)
        
        # Determina o resultado geral
        resultado_geral = validacao_ia.get("informacoes_corretas", False) and validacao_ia.get("confianca", 0) >= 0.8
        
        return {
            "mensagem_original": resposta,
            "relevante": True,
            "validacao_ia": validacao_ia,
            "resultado_geral": resultado_geral,
            "observacoes": validacao_ia.get("observacoes", "")
        }
    
    def validar_conversa(self, texto_conversa: str) -> List[Dict[str, Any]]:
        """Valida todas as mensagens do chatbot em uma conversa."""
        mensagens = self.extrair_mensagens_chatbot(texto_conversa)
        resultados = [self.validar_resposta(msg) for msg in mensagens]
        return resultados


# Função para teste rápido do validador
def demonstrar_validacao(conversa: str, api_key: str = None):
    """Demonstra o uso do validador de IA com uma conversa de exemplo."""
    validator = ChatbotIAValidator(api_key=api_key)
    resultados = validator.validar_conversa(conversa)
    
    print(f"\nAnalisando conversa com {len(resultados)} mens
          agens do chatbot:")
    for i, resultado in enumerate(resultados, 1):
        if not resultado["relevante"]:
            print(f"\nMensagem {i}: Não relevante para validação.")
            continue
            
        print(f"\nMensagem {i}: {resultado['mensagem_original']}")
        print(f"Resultado: {'VÁLIDO' if resultado['resultado_geral'] else 'INVÁLIDO'}")
        
        if resultado["validacao_ia"]:
            validacao = resultado["validacao_ia"]
            print(f"Confiança: {validacao.get('confianca', 0)*100:.1f}%")
            print(f"Informações presentes:")
            print(f"  - Tipo de exame: {'✓' if validacao.get('tipo_exame_presente', False) else '✗'}")
            print(f"  - Data de realização: {'✓' if validacao.get('data_realizacao_presente', False) else '✗'}")
            print(f"  - Data do laudo: {'✓' if validacao.get('data_laudo_presente', False) else '✗'}")
        
        print(f"Observações: {resultado['observacoes']}")


# Exemplo de uso
if __name__ == "__main__":
    # Exemplo de conversa do WhatsApp
    exemplo_conversa = """[15:26, 06/02/2025] Jonas: olá
[15:26, 06/02/2025] Futurotec Homologação: Olá, Jonas! Boa Tarde! Como posso ajudar?
[15:27, 06/02/2025] Jonas: gostaria de listar meus exames
[15:27, 06/02/2025] Futurotec Homologação: Olá, Jonas! Você tem um exame de ULTRASSONOGRAFIA realizado em 31/01/2025. O laudo está previsto para 03/02/2025. Como posso ajudar hoje?
[15:27, 06/02/2025] Futurotec Homologação: Você tem um exame de ULTRASSONOGRAFIA realizado em 31/01/2025. O laudo está previsto para 03/02/2025. Deseja saber mais sobre este exame ou agendar algo novo?"""
    
    # Variação para testar robustez
    variacao_conversa = """[15:26, 06/02/2025] Jonas: olá
[15:26, 06/02/2025] Futurotec Homologação: Olá, Jonas! Boa Tarde! Como posso ajudar?
[15:27, 06/02/2025] Jonas: gostaria de listar meus exames
[15:27, 06/02/2025] Futurotec Homologação: Olá, Jonas! Encontrei um exame de ULTRASSONOGRAFIA realizado em 31/01/2025. Seu laudo está previsto para 03/02/2025. Posso ajudar com mais alguma coisa?
[15:27, 06/02/2025] Futurotec Homologação: Você realizou um exame de ULTRASSONOGRAFIA em 31/01/2025 e o laudo será disponibilizado em 03/02/2025. Precisa de mais informações?"""
    
    # Exemplo com variação mais extrema de texto
    variacao_extrema = """[15:26, 06/02/2025] Jonas: olá
[15:26, 06/02/2025] Futurotec Homologação: Olá, Jonas! Boa Tarde! Como posso ajudar?
[15:27, 06/02/2025] Jonas: gostaria de listar meus exames
[15:27, 06/02/2025] Futurotec Homologação: Bom dia Jonas! Verifiquei que você possui um ultrassom que foi feito no final de janeiro, dia 31/01. O resultado do seu exame ficará disponível após o dia 3 de fevereiro.
[15:27, 06/02/2025] Futurotec Homologação: Consultei seu histórico e identifiquei um exame ultrassonográfico do dia 31 de janeiro deste ano. O resultado será liberado em 03/02. Posso ajudar com mais alguma coisa?"""
    
    # Substitua None pela sua chave de API se estiver usando uma API real
    api_key = None  # "sua-chave-api-aqui"
    
    print("=== VALIDAÇÃO DO EXEMPLO ORIGINAL ===")
    demonstrar_validacao(exemplo_conversa, api_key)
    
    print("\n\n=== VALIDAÇÃO DA VARIAÇÃO DE TEXTO ===")
    demonstrar_validacao(variacao_conversa, api_key)
    
    print("\n\n=== VALIDAÇÃO COM VARIAÇÃO EXTREMA DE TEXTO ===")
    demonstrar_validacao(variacao_extrema, api_key)
    
# PRECISO RODAR O DEEPSEEK LOCALMENTE?
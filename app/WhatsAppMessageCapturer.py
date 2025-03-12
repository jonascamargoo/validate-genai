from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import os

class WhatsAppMessageCapturer:
    def __init__(self, porta_debug=9222):
        """
        Conecta-se a uma instância do Chrome já aberta em modo de debug
        
        Args:
            porta_debug (int): Porta usada para remote debugging
        """
        print("aqui 1")
        # Configurações para conectar ao Chrome em execução
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{porta_debug}")
        print("aqui 2")

        # Caminho para o ChromeDriver (pode variar)
        
        service = Service(ChromeDriverManager().install())
        print("aqui 3")
        self.driver = webdriver.Chrome(service=service, options=chrome_options)
        print("aqui 3")
        
    def capturar_ultima_mensagem(self, contato):
        """
        Captura a última mensagem de um contato específico
        
        Args:
            contato (str): Nome do contato no WhatsApp
        
        Returns:
            str: Texto da última mensagem
        """
        try:
            # Localiza e clica no contato
            contato_elemento = WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.XPATH, f"//span[@title='{contato}']"))
            )
            contato_elemento.click()
            
            # Captura todas as mensagens
            mensagens = self.driver.find_elements_by_css_selector("div.message-text")
            
            # Retorna a última mensagem
            if mensagens:
                return mensagens[-1].text
            print("finalizou captura de mensagem")
            
            return None
        
        except Exception as e:
            print(f"Erro ao capturar mensagem: {e}")
            print("deu ruim na captura")

            return None

# Função para iniciar o Chrome em modo de debug
def iniciar_chrome_debug(porta=9222):
    """
    Inicia o Chrome em modo de debug
    
    Args:
        porta (int): Porta para remote debugging
    
    Returns:
        str: Comando para iniciar o Chrome
    """
    comando = (
        f'start chrome '
        f'--remote-debugging-port={porta} '
        f'--no-first-run '
        f'--no-default-browser-check '
        f'https://web.whatsapp.com'
    )
    
    # No Windows
    if os.name == 'nt':
        os.system(comando)
    # No Linux/Mac (adapte conforme necessário)
    else:
        print("??????????????")
        os.system(f'google-chrome --remote-debugging-port={porta} https://web.whatsapp.com')
    
    print(f"Chrome iniciado em modo de debug na porta {porta}")


# Exemplo de uso
def main():
    # Inicia Chrome em modo debug (opcional, se já não estiver aberto)
    iniciar_chrome_debug()
    print("Testando 1")
    # Conecta ao navegador
    capturador = WhatsAppMessageCapturer()
    print("testando 2")
    # Captura mensagem de um contato específico
    mensagem = capturador.capturar_ultima_mensagem("Jonas Camargo")
    print("testando 3")
    if mensagem:
        print("Última mensagem:", mensagem)

if __name__ == "__main__":
    main()
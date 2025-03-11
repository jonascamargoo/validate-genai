from selenium import webdriver
from selenium.webdriver.chrome.options import Options
import os

class WhatsAppMessageCapturer:
    def __init__(self, porta_debug=9222):
        """
        Conecta-se a uma instância do Chrome já aberta em modo de debug
        
        Args:
            porta_debug (int): Porta usada para remote debugging
        """
        # Configurações para conectar ao Chrome em execução
        chrome_options = Options()
        chrome_options.add_experimental_option("debuggerAddress", f"127.0.0.1:{porta_debug}")
        
        # Caminho para o ChromeDriver (pode variar)
        chrome_driver_path = "/path/to/chromedriver"
        
        # Conecta ao navegador já aberto
        self.driver = webdriver.Chrome(
            executable_path=chrome_driver_path, 
            options=chrome_options
        )
    
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
            contato_elemento = self.driver.find_element_by_xpath(f"//span[@title='{contato}']")
            contato_elemento.click()
            
            # Captura todas as mensagens
            mensagens = self.driver.find_elements_by_css_selector("div.message-text")
            
            # Retorna a última mensagem
            if mensagens:
                return mensagens[-1].text
            
            return None
        
        except Exception as e:
            print(f"Erro ao capturar mensagem: {e}")
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
        os.system(f'google-chrome --remote-debugging-port={porta} https://web.whatsapp.com')
    
    print(f"Chrome iniciado em modo de debug na porta {porta}")

# Exemplo de uso
def main():
    # Inicia Chrome em modo debug (opcional, se já não estiver aberto)
    iniciar_chrome_debug()
    
    # Conecta ao navegador
    capturador = WhatsAppMessageCapturer()
    
    # Captura mensagem de um contato específico
    mensagem = capturador.capturar_ultima_mensagem("Nome do Contato")
    
    if mensagem:
        print("Última mensagem:", mensagem)

if __name__ == "__main__":
    main()
import unittest

from pre_processor import (
    remove_urls,
    remove_special_characters,
    remove_numbers,
    tokenize_text,
    remove_stopwords_and_short_tokens,
    apply_stemming,
    pre_process_portuguese
)

class TestPreProcessor(unittest.TestCase):
    def test_basic_text(self):
        """Testa se o texto básico é processado corretamente."""
        text = "Hoje é um ótimo dia para estudar!"
        expected = "hoj ótim dia estud"  # Exemplo do resultado esperado após stemming
        self.assertEqual(pre_process_portuguese(text), expected)

    def test_text_with_numbers(self):
        """Testa se números são removidos corretamente."""
        text = "Tenho 2 gatos e 3 cachorros."
        expected = "Tenho  gatos e  cachorros."
        self.assertEqual(remove_numbers(text), expected)

    def test_text_with_url(self):
        """Testa se URLs são removidas corretamente."""
        text = "Veja isso em http://exemplo.com"
        expected = "Veja isso em "  # Se a URL era o único conteúdo, o texto final deve ser vazio
        self.assertEqual(remove_urls(text), expected)

    def test_text_with_special_characters(self):
        """Testa se caracteres especiais são removidos."""
        text = "Olá, mundo! #feliz :)"
        expected = "Olá mundo feliz "  # "Olá" vira "olá" e depois "ol" pelo stemmer
        self.assertEqual(remove_special_characters(text), expected)
        
    def test_tokenize_text(self):
        text = "Olá, mundo! Como você está?"
        tokens = tokenize_text(text)
        expected_tokens = ['Olá', ',', 'mundo', '!', 'Como', 'você', 'está', '?']
        self.assertEqual(tokens, expected_tokens)

    def test_remove_stopwords_and_short_tokens(self):
        tokens = ['Como', 'você', 'está', 'a', 'poder']
        filtered_tokens = remove_stopwords_and_short_tokens(tokens)
        expected_tokens = ['Como', 'poder']
        self.assertEqual(filtered_tokens, expected_tokens)

    def test_apply_stemming(self):
        tokens = ['comprei', 'comprando', 'compra', 'poder', 'jogando']
        stemmed_tokens = apply_stemming(tokens)
        expected_tokens = ['compr', 'compr', 'compr', 'pod', 'jog']
        self.assertEqual(stemmed_tokens, expected_tokens)

    def test_empty_string(self):
        """Testa se uma string vazia retorna vazia."""
        self.assertEqual(pre_process_portuguese(""), "")

    def test_non_string_input(self):
        """Testa se entrada não string retorna vazia."""
        self.assertEqual(pre_process_portuguese(12345), "")

if __name__ == "__main__":
    unittest.main()

import unittest
import nltk
from src.utils.tokenizer import Tokenizer


class TestTokenizer(unittest.TestCase):

    texts = [
        'едва не з..........аорал:""ебать-копать! Это же снег!""',
        'Да как же он ЗА*БАЛ!!!',
        'Читай по губам: пи да рас)',
        'после этого он стал сам не свой.почему - хз',
        'x\ne\nr\n',
        'из-за тебя что - то пошло не по плану',
        'смотри шутку: 3.14дор',
        'на🤬 возьми',
    ]

    def test_tokenize(self):
        print('-' * 50 + '\nBase Tokenizer\n' + '-' * 50)
        tokenizer = Tokenizer()

        for text in self.texts:
            print('|'.join(tokenizer.tokenize(text)))

        print('-' * 50 + '\nNLTK  WordPunctTokenizer\n' + '-' * 50)
        tokenizer = nltk.WordPunctTokenizer()

        for text in self.texts:
            print('|'.join(tokenizer.tokenize(text)))

        print('-' * 50 + '\nNLTK  TreebankWordTokenizer\n' + '-' * 50)
        tokenizer = nltk.TreebankWordTokenizer()

        for text in self.texts:
            print('|'.join(tokenizer.tokenize(text)))

        self.assertTrue(True)

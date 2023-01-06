class Tokenizer:
    """
    Класс для токенизации предложения на токены
    """
    def __init__(self):
        pass

    @staticmethod
    def tokenize(text: str) -> list:
        """
        Метод для разбиения стоки на токены
        :param text: исходная строка
        :return: список токенов
        """
        return text.split()

    @staticmethod
    def detokenize(tokens: list) -> str:
        """
        Метод для детокинезации
        :param tokens: список токенов
        :return: цельная строка
        """
        return ' '.join(tokens)

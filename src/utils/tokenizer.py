class Tokenizer:
    def __init__(self):
        pass

    @staticmethod
    def tokenize(text: str) -> list:
        return text.split()

    @staticmethod
    def detokenize(tokens: list) -> str:
        return ' '.join(tokens)

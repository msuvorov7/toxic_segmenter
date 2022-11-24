import re


class Preprocessor:
    def __init__(self):
        # self.noise_char_reg = re.compile(r'[.,!?"()-_—:;«»=±+#%&*№1234567890\\/\[\]|⁉~]')
        self.noise_char_reg = re.compile('[^a-zA-Zа-яА-ЯЁё]')
        self.emoji_pattern = re.compile(
            "["
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F700-\U0001F77F"  # alchemical symbols
            "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
            "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002702-\U000027B0"  # Dingbats
            "\U000024C2-\U0001F251"
            "]+"
        )
        self.translit_dict = {
            'a': 'а',
            'b': 'б',
            'c': 'с',
            'd': 'д',
            'e': 'е',
            'f': 'а',
            'g': 'г',
            'h': 'х',
            'i': 'и',
            'j': 'й',
            'k': 'к',
            'l': 'л',
            'm': 'м',
            'n': 'н',
            'o': 'о',
            'p': 'п',
            'q': 'й',
            'r': 'р',
            's': 'с',
            't': 'т',
            'u': 'у',
            'v': 'в',
            'w': 'ш',
            'x': 'х',
            'y': 'у',
            'z': 'з',
        }

    def remove_noise(self, token: str) -> str:
        return self.noise_char_reg.sub('', token)

    def remove_emoji(self, token: str) -> str:
        return self.emoji_pattern.sub(r'', token)

    @staticmethod
    def remove_duplicated_chars(token: str) -> str:
        res = '\r'
        for char in token:
            if char != res[-1]:
                res += char
        return res[1:]

    @staticmethod
    def replace_symbol_to_ru(token: str) -> str:
        token = token.replace('3.147', 'пи')
        token = token.replace('3,147', 'пи')
        token = token.replace('3.14', 'пи')
        token = token.replace('3,14', 'пи')
        token = token.replace('0', 'о')
        token = token.replace('@', 'а')
        token = token.replace('$', 'с')
        token = token.replace('}{', 'х')
        token = token.replace('1', 'л')
        token = token.replace('4', 'ч')
        token = token.replace('’', 'ь')
        token = token.replace('ё', 'е')
        token = token.replace('🆘', 'sos')
        return token

    def translit(self, token: str) -> str:
        for char in token:
            if char in self.translit_dict:
                token = token.replace(char, self.translit_dict[char])
        return token

    def forward(self, token: str) -> str:
        lower_token = token.lower()
        lower_token = self.replace_symbol_to_ru(lower_token)
        # lower_token = self.remove_emoji(lower_token)
        lower_token = self.remove_noise(lower_token)
        lower_token = self.remove_duplicated_chars(lower_token)
        # lower_token = self.translit(lower_token)

        return lower_token

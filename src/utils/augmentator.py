import random

import numpy as np


class Augmentator:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.noise_string = "qwertyuiop[]asdfghjkl;'\|`~zxcvbnm,./?<>§±1234567890-=!@#$%^&*(" \
                            ")_+№%:йцукенгшщзхъфывапролджэё][ячсмитьбю "
        self.ru_2_lat = {
            'а': ['a', '@'],
            'б': ['b', '6'],
            'в': ['v'],
            'г': ['g'],
            'д': ['d', 'g'],
            'е': ['e'],
            'ё': ['e'],
            'ж': ['zh'],
            'з': ['z', '3'],
            'и': ['i'],
            'й': ['i'],
            'к': ['k'],
            'л': ['l', '1', '|'],
            'м': ['m'],
            'н': ['n'],
            'о': ['o', '0'],
            'п': ['p', 'n'],
            'р': ['r'],
            'с': ['s', '$', 'c'],
            'т': ['t'],
            'у': ['u', 'y'],
            'ф': ['f'],
            'х': ['h', 'x', '}{'],
            'ц': ['c'],
            'ч': ['cz', '4', 'ch'],
            'ш': ['sh'],
            'щ': ['scz'],
            'ъ': [''],
            'ы': ['y'],
            'ь': ['b', '’'],
            'э': ['e'],
            'ю': ['u'],
            'я': ['ja'],
        }
        self.replace_grammar = {
            'а': 'о',
            'о': 'а',
            'е': 'и',
            'и': 'е',
            'м': 'н',
            'н': 'м',
            'в': 'ф',
            'ф': 'в',
            'у': 'ю',
            'ю': 'у',
            'б': 'п',
            'я': 'е',
        }

    def _is_empty_token(self, token: str) -> bool:
        return len(token) == 0

    def randomly_remove(self, token: str) -> str:
        if self._is_empty_token(token):
            return token

        p = np.random.rand()

        if p <= self.probability:
            index_to_remove = np.random.choice(np.arange(len(token)))
            return token[:index_to_remove] + token[index_to_remove + 1:]
        return token

    def randomly_noise(self, token: str) -> str:
        if self._is_empty_token(token):
            return token

        p = np.random.rand()

        if p <= self.probability:
            index_to_paste = np.random.choice(np.arange(len(token)))
            index_to_noise = np.random.choice(np.arange(len(self.noise_string)))
            count_of_noise = np.random.choice(np.arange(3))
            return (token[:index_to_paste]
                    + self.noise_string[index_to_noise] * count_of_noise
                    + token[index_to_paste:])
        return token

    def randomly_replace_to_latin(self, token: str) -> str:
        if self._is_empty_token(token):
            return token

        p = np.random.rand()
        if p <= self.probability:
            replace_num = np.random.choice(np.array(len(token))) // 2
            for _ in range(replace_num):
                char = random.choice(list(token))
                if char in self.ru_2_lat:
                    latin = random.choice(list(self.ru_2_lat[char]))
                    token = token.replace(char, latin)
        return token

    def randomly_replace_grammar(self, token: str) -> str:
        if self._is_empty_token(token):
            return token

        p = np.random.rand()
        if p <= self.probability:
            for _ in range(2):
                char = random.choice(list(self.replace_grammar.keys()))
                token = token.replace(char, self.replace_grammar[char])
        return token

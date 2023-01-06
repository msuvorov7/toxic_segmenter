import unittest

import compress_fasttext
import onnxruntime

from src.utils.transformer import FeatureTransformer


class TestPredict(unittest.TestCase):
    fasttext_model = compress_fasttext.models.CompressedFastTextKeyedVectors.load('../models/tiny_fasttext.model')
    model = onnxruntime.InferenceSession('../models/segmenter.onnx')
    transformer = FeatureTransformer(fasttext_model, model)

    def test_base_predict(self):
        texts = [
            'пидрила злоебучий убери свою смазливую морду.',
            'собака конченная вот ты кто.',
            'знаю я породу этих хуеплетов.',
            'копать не строить.',
            'мне кажется этот пидарок слишком драмматизирует.',
            'еб@нько прикрой, пидрк.',
            'пиздацирк какой-то.',
            'мазь и словарь проверь.',
            'смотри, как он упиздячивает!',
            'а герой снова гондон',
        ]

        toxic_smile = '🤬'
        threshold = 0.2

        for text in texts:

            result_text = ''
            tokens = self.transformer.tokenizer.tokenize(text)
            probabilities = self.transformer.predict(text)

            for tok, prob in zip(tokens, probabilities):
                result_text += f'{tok}({prob:.2f}){toxic_smile} ' if prob > threshold else f'{tok}({prob:.2f}) '

            print(result_text)

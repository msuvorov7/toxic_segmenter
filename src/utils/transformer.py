import numpy as np

from src.utils.preprocess_rules import Preprocessor
from src.utils.tokenizer import Tokenizer


class FeatureTransformer:
    def __init__(self, fasttext_model, onnx_model):
        self.tokenizer = Tokenizer()
        self.preprocessor = Preprocessor()
        self.fasttext_model = fasttext_model
        self.model = onnx_model

    def fit(self):
        return self

    @staticmethod
    def softmax(z: np.ndarray):
        exp = np.exp(z - np.max(z))
        for i in range(len(z)):
            exp[i] /= np.sum(exp[i])
        return exp

    def predict(self, message: str) -> np.ndarray:
        tokens = self.tokenizer.tokenize(message)
        processed_tokens = [self.preprocessor.forward(token) for token in tokens]
        encoded = [self.fasttext_model[item] for item in processed_tokens]

        model_input = {self.model.get_inputs()[0].name: encoded}
        model_output = self.model.run(None, model_input)
        probabilities = self.softmax(model_output[0][0])[:, 1]
        return probabilities

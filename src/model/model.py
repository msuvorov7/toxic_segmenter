import torch.nn as nn


class ToxicSegmenter(nn.Module):
    def __init__(
            self,
            embedding_dim: int,
            hidden_size: int,
            output_dim: int,
            dropout=0.5,
    ):
        super().__init__()

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=1,
                            batch_first=True,
                            bidirectional=True)

        self.fc = nn.Linear(2 * hidden_size, output_dim)

    def forward(self, text):
        x, (_, _) = self.lstm(text)

        return self.fc(x)
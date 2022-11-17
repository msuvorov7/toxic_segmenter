import torch.nn as nn


class ToxicSegmenter(nn.Module):
    """
    Модель для классификации токсичных токенов
    """
    def __init__(
            self,
            embedding_dim: int,
            hidden_size: int,
            output_dim: int,
            dropout=0.5,
            is_bidirectional: bool = True,
    ):
        super().__init__()

        self.lstm = nn.LSTM(input_size=embedding_dim,
                            hidden_size=hidden_size,
                            num_layers=3,
                            batch_first=True,
                            bidirectional=is_bidirectional)

        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=is_bidirectional
                          )

        self.dropout = nn.Dropout(p=dropout)

        self.fc = nn.Linear((int(is_bidirectional) + 1) * hidden_size, output_dim)

    def forward(self, text):
        if len(text.shape) == 2:
            text = text.view(1, text.shape[0], text.shape[1])
        # x, (_, _) = self.lstm(text)
        # return self.fc(x)
        x, hidden = self.gru(text)
        return self.fc(self.dropout(x))

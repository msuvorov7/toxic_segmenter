import torch.nn as nn
import torch.nn.functional as F


class ToxicSegmenter(nn.Module):
    """
    Модель для классификации токсичных токенов
    """
    def __init__(
            self,
            embedding_dim: int,
            hidden_size: int,
            output_dim: int,
            dropout=0.2,
            is_bidirectional: bool = True,
    ):
        super().__init__()

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

        x, hidden = self.gru(text)
        x = F.relu(x)

        return self.fc(x)

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
            dropout=0.5,
            is_bidirectional: bool = False,
    ):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.conv = nn.Conv1d(in_channels=embedding_dim,
                              out_channels=hidden_size,
                              kernel_size=(3,),
                              padding=(1,)
                              )

        self.weights = nn.Linear(in_features=embedding_dim, out_features=embedding_dim)

        self.gru = nn.GRU(input_size=embedding_dim,
                          hidden_size=hidden_size,
                          num_layers=1,
                          batch_first=True,
                          bidirectional=is_bidirectional
                          )

        self.fc = nn.Linear((int(is_bidirectional) + 1) * hidden_size, output_dim)

    def forward(self, text):
        # print(text.shape)
        if len(text.shape) == 2:
            text = text.view(1, text.shape[0], text.shape[1])
        # print(text.shape)
        # print(text.shape)
        # print(self.weights(text).shape)
        conved = self.conv(text.permute(0, 2, 1))
        conved = F.relu(conved)
        # print(pooled.shape)
        # x, hidden = self.gru(text)
        return self.fc(conved.permute(0, 2, 1))

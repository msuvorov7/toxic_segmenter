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

        self.conv_1 = nn.Conv1d(in_channels=embedding_dim,
                                out_channels=hidden_size,
                                kernel_size=(3,),
                                padding=(1,)
                                )

        self.polling = nn.MaxPool1d(3, padding=1)

        self.conv_2 = nn.Conv1d(in_channels=86,
                                out_channels=128,
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

        self.fc = nn.Linear((int(is_bidirectional) + 1) * 128, output_dim)

    def forward(self, text):
        # print(text.shape)
        if len(text.shape) == 2:
            text = text.view(1, text.shape[0], text.shape[1])
        # print('text: ', text.shape)
        # print(self.weights(text).shape)
        conved = self.conv_1(text.permute(0, 2, 1))
        # print('conv_1: ', conved.shape)
        pooled = self.polling(conved.permute(0, 2, 1))
        # print('pool: ', pooled.shape)
        conved = self.conv_2(pooled.permute(0, 2, 1))
        # print('conv_2: ', conved.shape)
        x = F.relu(conved.permute(0, 2, 1))

        # x, hidden = self.gru(text)
        return self.fc(x)

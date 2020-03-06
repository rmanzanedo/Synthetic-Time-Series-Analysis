# import torch
import torch.nn as nn


class rnn(nn.Module):

    def __init__(self, input_size=253, hidden_size=512):
        super(rnn, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=2, dropout=0.5)
        self.fc = nn.Sequential(
            nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, 2),
            nn.Softmax(1)
        )

    def forward(self, x):
        out, (hn, cn) = self.rnn(x, None)
        out = self.fc(hn[-1])

        return out, hn[-1]
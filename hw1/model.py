from typing import Dict

import torch
import torch.nn as nn

class SeqClassifier(torch.nn.Module):
    def __init__(
        self,
        embeddings: torch.tensor,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        bidirectional: bool,
        num_class: int,
        seq_len: int,
    ) -> None:
        super(SeqClassifier, self).__init__()
        self.embed = nn.Embedding.from_pretrained(embeddings, freeze=False)
        # TODO: model architecture
        self.lstm = nn.LSTM(input_size=input_size,hidden_size=hidden_size,num_layers=num_layers,dropout=dropout,bidirectional=bidirectional,batch_first=True)

        self.feature_size = seq_len * hidden_size
        if bidirectional:
            self.feature_size *= 2

        self.fc = nn.Sequential(
            nn.Linear(self.feature_size,1024),
            nn.BatchNorm1d(1024),
            # nn.LeakyReLU(0.2),
            nn.ReLU(),
            nn.Linear(1024,num_class),
            nn.Sigmoid()
        )


    #     # TODO: implement model forward
    def forward(self, input):
        embed = self.embed(input)
        output, (h_n, c_n) = self.lstm(embed)
        output = self.fc(output.contiguous().view(output.shape[0],-1))
        return output
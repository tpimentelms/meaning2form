import torch.nn as nn

from .base import BaseLM


class IpaLM(BaseLM):
    name = 'lstm'

    def __init__(self, vocab_size, hidden_size, nlayers=1, dropout=0.1, embedding_size=None, **kwargs):
        super().__init__(
            vocab_size, hidden_size, nlayers=nlayers, dropout=dropout, embedding_size=embedding_size, **kwargs)

        self.embedding = nn.Embedding(vocab_size, self.embedding_size)
        self.lstm = nn.LSTM(
            self.embedding_size, hidden_size, nlayers, dropout=(dropout if nlayers > 1 else 0), batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, idx):
        h_old = self.context(idx)
        x_emb = self.dropout(self.get_embedding(x))

        c_t, h_t = self.lstm(x_emb, h_old)
        c_t = self.dropout(c_t).contiguous()

        logits = self.out(c_t)
        return logits, h_t

    def get_embedding(self, x):
        return self.embedding(x)

    def initHidden(self, bsz=1):
        weight = next(self.parameters()).data
        return weight.new(self.nlayers, bsz, self.hidden_size).zero_(), \
            weight.new(self.nlayers, bsz, self.hidden_size).zero_()

from abc import abstractmethod
import torch.nn as nn


class Word2Vec(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int, max_norm: int):
        super(Word2Vec, self).__init__()
        self.embeddings = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            max_norm=max_norm,
        )
        self.linear = nn.Linear(
            in_features=embedding_dim,
            out_features=vocab_size,
        )

    @abstractmethod
    def forward(self, inputs):
        pass


class CBOWModel(Word2Vec):
    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = out.mean(axis=1)
        out = self.linear(out)
        return out


class SkipGramModel(Word2Vec):
    def forward(self, inputs):
        out = self.embeddings(inputs)
        out = self.linear(out)
        return out

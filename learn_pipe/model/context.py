import torch
import torch.nn as nn
import pickle
import math
from sklearn.decomposition import PCA

from data_pipe.northeuralex import NortheuralexInfo
from data_pipe.celex import CelexInfo


class Context(nn.Module):
    def __init__(self, hidden_size, nlayers=1):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size


class NoContext(Context):
    def forward(self, x):
        bsz = x.size(0)
        return x.new(self.nlayers, bsz, self.hidden_size).zero_().float(), \
            x.new(self.nlayers, bsz, self.hidden_size).zero_().float()


class POSContext(Context):
    def __init__(self, hidden_size, data, nlayers=1, dropout=0.1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.get_pos_embs(hidden_size, nlayers, data=data)

    def get_pos_embs(self, hidden_size, nlayers, data):
        if data == 'northeuralex':
            self.data_info = NortheuralexInfo()
        elif data == 'celex':
            self.data_info = CelexInfo()
        else:
            raise ValueError('Word2VecContext not implemented for data %s' % data)

        pos_tensor = torch.Tensor(self.data_info.pos).long()
        self.pos = nn.Parameter(pos_tensor, requires_grad=False)
        self.c_embedding = nn.Embedding(self.data_info.npos, hidden_size * nlayers)
        self.h_embedding = nn.Embedding(self.data_info.npos, hidden_size * nlayers)

    def forward(self, x):
        return self._pos_forward(x)

    def _pos_forward(self, x):
        bsz = x.size(0)

        x = self.pos[x]
        x_c_emb = self.dropout(
            self.c_embedding(x).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous())
        x_h_emb = self.dropout(
            self.h_embedding(x).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous())

        return x_c_emb, x_h_emb


class Word2VecContext(Context):
    def __init__(self, hidden_size, data, word2vec_size=10, nlayers=1, dropout=0.1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.word2vec_size = word2vec_size
        self.data = data
        self.get_embs(hidden_size, word2vec_size, nlayers, data=data)

    def get_embs(self, hidden_size, word2vec_size, nlayers, data='northeuralex', base_ffolder='datasets/'):
        fname = '%s/%s/filtered-word2vec.pckl' % (base_ffolder, data)
        with open(fname, 'rb') as f:
            self.word2vec = pickle.load(f)
        self.word2vec = {k: x for k, x in self.word2vec.items() if x is not None}

        if data == 'northeuralex':
            self.data_info = NortheuralexInfo()
            idx2context, context_vecs = self.data_info.build_vec_matrix(self.word2vec)
        elif data == 'celex':
            self.data_info = CelexInfo()
            idx2context, context_vecs = self.data_info.build_vec_matrix(self.word2vec)
        else:
            raise ValueError('Word2VecContext not implemented for data %s' % data)

        idx2context_tensor = torch.Tensor(idx2context).long()
        self.idx2context = nn.Parameter(idx2context_tensor, requires_grad=False)

        pca = PCA(n_components=word2vec_size)
        context_vecs = pca.fit_transform(context_vecs)

        self.c_embedding = nn.Embedding(context_vecs.shape[0], context_vecs.shape[1])
        self.c_embedding.weight.data.copy_(nn.Parameter(torch.from_numpy(context_vecs), requires_grad=False))
        self.c_embedding.weight.requires_grad = False
        self.h_embedding = nn.Embedding(context_vecs.shape[0], context_vecs.shape[1])
        self.h_embedding.weight.data.copy_(nn.Parameter(torch.from_numpy(context_vecs), requires_grad=False))
        self.h_embedding.weight.requires_grad = False

        self.c_linear = nn.Linear(context_vecs.shape[1], hidden_size * nlayers)
        self.h_linear = nn.Linear(context_vecs.shape[1], hidden_size * nlayers)

    def forward(self, x):
        return self._w2v_forward(x)

    def _w2v_forward(self, x):
        bsz = x.size(0)
        x = self.idx2context[x]
        assert (x != -1).all()

        x_c_emb = self.dropout(self.c_embedding(x))
        x_c_emb = self.dropout(
            self.c_linear(x_c_emb).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous())
        x_h_emb = self.dropout(self.h_embedding(x))
        x_h_emb = self.dropout(
            self.h_linear(x_h_emb).reshape(bsz, self.nlayers, -1).transpose(0, 1).contiguous())

        return x_c_emb, x_h_emb


class MixedContext(Context):
    def __init__(self, hidden_size, data, word2vec_size=10, nlayers=1, dropout=0.1):
        super().__init__(hidden_size, nlayers=nlayers)
        self.dropout = nn.Dropout(dropout)
        self.pos = POSContext(math.ceil(hidden_size / 2), data, nlayers, dropout=dropout)
        self.w2v = Word2VecContext(math.floor(hidden_size / 2), data, word2vec_size, nlayers, dropout=dropout)

    def forward(self, x):
        pos_c_embs, pos_h_embs = self.pos(x)
        w2v_c_embs, w2v_h_embs = self.w2v(x)

        x_c_embs = torch.cat([pos_c_embs, w2v_c_embs], dim=-1)
        x_h_embs = torch.cat([pos_h_embs, w2v_h_embs], dim=-1)

        return x_c_embs, x_h_embs

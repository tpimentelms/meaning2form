import copy
import torch
import torch.nn as nn

from .context import \
    NoContext, POSContext, \
    Word2VecContext, MixedContext
from util import constants


class BaseLM(nn.Module):
    name = 'base'

    def __init__(
            self, vocab_size, hidden_size, nlayers=1, dropout=0.1,
            embedding_size=None, word2vec_size=10, context=None, data=None):
        super().__init__()
        self.nlayers = nlayers
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size if embedding_size is not None else hidden_size
        self.word2vec_size = word2vec_size
        self.dropout_p = dropout
        self.vocab_size = vocab_size

        self.best_state_dict = None
        self.data = data
        self.load_context(context)

    def load_context(self, context):
        self.context_type = context
        if context is None or context == 'none':
            self.context = NoContext(self.hidden_size, nlayers=self.nlayers)
        elif context == 'pos':
            self.context = POSContext(self.hidden_size, nlayers=self.nlayers, dropout=self.dropout_p, data=self.data)
        elif context == 'word2vec':
            assert self.data is not None, 'data can not be None for word2vec context'
            self.context = Word2VecContext(
                self.hidden_size, word2vec_size=self.word2vec_size, nlayers=self.nlayers,
                dropout=self.dropout_p, data=self.data)
        elif context == 'mixed':
            assert self.data is not None, 'data can not be None for mixed context'
            self.context = MixedContext(
                self.hidden_size, word2vec_size=self.word2vec_size, nlayers=self.nlayers,
                dropout=self.dropout_p, data=self.data)
        else:
            raise ValueError('Invalid context name %s' % context)

    def set_best(self):
        self.best_state_dict = copy.deepcopy(self.state_dict())

    def recover_best(self):
        self.load_state_dict(self.best_state_dict)

    def save(self, path, context, suffix):
        fname = self.get_name(path, context, suffix)
        torch.save({
            'kwargs': self.get_args(),
            'model_state_dict': self.state_dict(),
        }, fname)

    def get_args(self):
        return {
            'nlayers': self.nlayers,
            'hidden_size': self.hidden_size,
            'embedding_size': self.embedding_size,
            'dropout': self.dropout_p,
            'vocab_size': self.vocab_size,
            'context': self.context_type,
            'data': self.data,
        }

    @classmethod
    def load(cls, path, suffix):
        checkpoints = cls.load_checkpoint(path, suffix)
        model = cls(**checkpoints['kwargs'])
        model.load_state_dict(checkpoints['model_state_dict'])
        return model

    @classmethod
    def load_checkpoint(cls, path, context, suffix):
        fname = cls.get_name(path, context, suffix)
        return torch.load(fname, map_location=constants.device)

    @classmethod
    def get_name(cls, path, context, suffix):
        return '%s/%s__%s__%s.tch' % (path, cls.name, context, suffix)

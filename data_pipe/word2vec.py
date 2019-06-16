# import pandas as pd
# import numpy as np
import pickle


class Word2VecInfo(object):
    def __init__(self, ffolder='datasets/northeuralex/', drop_none=True):
        self.word2vec = self.get_word2vec(ffolder=ffolder, drop_none=drop_none)

    @staticmethod
    def get_word2vec(ffolder='datasets/', drop_none=True):
        fname = '%s/filtered-word2vec.pckl' % (ffolder)
        with open(fname, 'rb') as f:
            word2vec = pickle.load(f)
        if drop_none:
            word2vec = {k: x for k, x in word2vec.items() if x is not None}
        return word2vec

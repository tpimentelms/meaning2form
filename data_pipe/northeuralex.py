import pandas as pd
import numpy as np


class NortheuralexInfo(object):
    def __init__(self, ffolder='datasets/northeuralex/'):
        self.df = self.get_df(ffolder=ffolder)

        self.get_parsed_pos()
        self.get_idx_to_context()

    def get_pos(self, index):
        return self.pos[index]

    @staticmethod
    def get_df(ffolder='datasets/northeuralex/'):
        filename = '%s/orig.csv' % ffolder
        df = pd.read_csv(filename, sep='\t')
        del df['Glottocode']
        df = df.dropna()
        df['new_id'] = range(df.shape[0])
        df['map_id'] = df.Concept_ID
        df['phoneme'] = df.IPA.apply(lambda x: x.split(' '))
        df.set_index('new_id', inplace=True)
        return df

    def get_parsed_pos(self):
        self.df['pos_base'] = self.df.Concept_ID.apply(lambda x: x.split(':')[-1])
        pos_tokens = sorted(set(self.df.pos_base.unique()))
        self.pos_map = {x: i for i, x in enumerate(pos_tokens)}
        self.npos = self.df.pos_base.unique().shape[0]

        self.df['pos'] = self.df.pos_base.apply(lambda x: self.pos_map[x])
        self.pos_dict = self.df.pos.to_dict()

        max_idx = self.df.index.max()

        self.pos = np.ones(max_idx + 1) * -1
        self.pos[list(self.pos_dict.keys())] = list(self.pos_dict.values())

    def get_idx_to_context(self):
        self.concept_tokens = sorted(set(self.df.Concept_ID.unique()))
        self.nconcepts = len(self.concept_tokens)
        self.idx2context = self.df.Concept_ID.to_dict()

    def build_vec_matrix(self, word2vec):
        self.vec_df = self.df[self.df.Concept_ID.isin(word2vec)]

        self.vec_concept_tokens = sorted(set(self.vec_df.Concept_ID.unique()))
        self.concept_map = {x: i for i, x in enumerate(self.vec_concept_tokens)}

        idx2context_dict = {idx: self.concept_map[row.Concept_ID] for idx, row in self.vec_df.iterrows()}

        max_idx = self.vec_df.index.max()
        self.idx2vec_context = np.ones(max_idx + 1) * -1
        self.idx2vec_context[list(idx2context_dict.keys())] = list(idx2context_dict.values())

        self.context_vecs = np.matrix([word2vec[x] for x in self.vec_concept_tokens])

        return self.idx2vec_context, self.context_vecs

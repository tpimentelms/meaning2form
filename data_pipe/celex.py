import pandas as pd
import numpy as np


class CelexInfo(object):
    def __init__(self, ffolder='datasets/celex/'):
        self.df = self.get_df(ffolder=ffolder)

        self.get_parsed_pos()
        self.get_idx_to_context()

    def get_pos(self, index):
        return self.pos[index]

    @classmethod
    def get_df(cls, ffolder='datasets/celex/', languages=['eng', 'deu', 'nld']):
        dfs = []
        for language in languages:
            dfs += [cls.read_src_data_single(ffolder, language)]
        df = pd.concat(dfs)
        df['new_id'] = range(df.shape[0])
        df['map_id'] = df.new_id
        df['phoneme'] = df.phon.apply(lambda x: [y for y in x])

        df.set_index('new_id', inplace=True)
        return df

    @staticmethod
    def read_src_data_single(ffolder, language):
        filename = '%s/info__lemma_%s_1_0_0_10000.tsv' % (ffolder, language)
        df = pd.read_csv(filename, sep='\t')
        df['Language_ID'] = language
        return df

    def get_parsed_pos(self):
        pos_tokens = sorted(set(self.df.grammar.unique()))
        self.pos_map = {x: i for i, x in enumerate(pos_tokens)}

        self.df['pos'] = self.df.grammar.apply(lambda x: self.pos_map[x])
        self.pos_dict = self.df.pos.to_dict()

        max_idx = self.df.index.max()

        self.pos = np.ones(max_idx + 1) * -1
        self.pos[list(self.pos_dict.keys())] = list(self.pos_dict.values())
        self.npos = self.df.grammar.unique().shape[0]

    def get_idx_to_context(self):
        self.concept_tokens = sorted(set([str(x) for x in self.df.word.unique()]))
        self.nconcepts = len(self.concept_tokens)
        self.idx2context = self.df.word.to_dict()

    def build_vec_matrix(self, word2vec):
        self.vec_df = self.df[self.df.index.isin(word2vec)]

        self.vec_ids = sorted(set(self.vec_df.index.unique()))
        self.id_map = {x: i for i, x in enumerate(self.vec_ids)}

        idx2context_dict = {idx: self.id_map[idx] for idx, row in self.vec_df.iterrows()}

        max_idx = self.vec_df.index.max()
        self.idx2vec_context = np.ones(max_idx + 1) * -1
        self.idx2vec_context[list(idx2context_dict.keys())] = list(idx2context_dict.values())

        self.context_vecs = np.matrix([word2vec[x] for x in self.vec_ids])

        return self.idx2vec_context, self.context_vecs

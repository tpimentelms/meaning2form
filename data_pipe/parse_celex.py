# Note: Dataset 0 is PAD 1 is SOW and 2 is EOW
import pandas as pd
import numpy as np
import pickle

import sys
sys.path.append('./')
import util.argparser as parser
from util import util
from data_pipe.word2vec import Word2VecInfo


def read_src_data_single(ffolder, language):
    filename = '%s/info__lemma_%s_1_0_0_10000.tsv' % (ffolder, language)
    df = pd.read_csv(filename, sep='\t')
    df['Language_ID'] = language
    return df


def read_src_data(ffolder, languages=['eng', 'deu', 'nld']):
    dfs = []
    for language in languages:
        dfs += [read_src_data_single(ffolder, language)]
    df = pd.concat(dfs)
    df['new_id'] = range(df.shape[0])
    return df


def get_languages():
    return ['eng', 'deu', 'nld']


def get_phrases(df):
    phrases = df.new_id.unique()
    np.random.shuffle(phrases)
    return list(phrases)


def separate_df(df, train_set, val_set, test_set):
    train_df = df[df['new_id'].isin(train_set)]
    val_df = df[df['new_id'].isin(val_set)]
    test_df = df[df['new_id'].isin(test_set)]

    return train_df, val_df, test_df


def separate_train_language(df, language):
    df_lang = df[df.Language_ID == language]
    phrases = get_phrases(df_lang)

    num_sentences = len(phrases)

    train_size = int(num_sentences * .8)
    val_size = int(num_sentences * .1)
    test_size = num_sentences - train_size - val_size

    train_set = phrases[:train_size]
    val_set = phrases[train_size:-test_size]
    test_set = phrases[-test_size:]

    return train_set, val_set, test_set


def separate_train(df):
    train_set, val_set, test_set = [], [], []
    for language in df.Language_ID.unique():
        train_set_single, val_set_single, test_set_single = separate_train_language(df, language)

        train_set += train_set_single
        val_set += val_set_single
        test_set += test_set_single

    data_split = (train_set, val_set, test_set)
    train_df, val_df, test_df = separate_df(df, train_set, val_set, test_set)
    return train_df, val_df, test_df, data_split


def separate_per_language(train_df, val_df, test_df, languages):
    languages_df_train = separate_per_language_single_df(train_df, languages)
    languages_df_val = separate_per_language_single_df(val_df, languages)
    languages_df_test = separate_per_language_single_df(test_df, languages)

    languages_df = {lang: {
        'train': languages_df_train[lang],
        'val': languages_df_val[lang],
        'test': languages_df_test[lang],
    } for lang in languages_df_train.keys()}
    return languages_df


def separate_per_language_single_df(df, languages):
    languages_df = {lang: df[df['Language_ID'] == lang] for lang in languages}
    return languages_df


def get_tokens(df):
    tokens = set()
    for index, x in df.iterrows():
        try:
            tokens |= set(x.phon)
        except:
            continue

    tokens = sorted(list(tokens))
    token_map = {x: i + 3 for i, x in enumerate(tokens)}
    token_map['PAD'] = 0
    token_map['SOW'] = 1
    token_map['EOW'] = 2

    return token_map


def process_languages(languages_df, token_map, args):
    util.mkdir('%s/preprocess%s/' % (args.ffolder, args.fsuffix))
    for lang, df in languages_df.items():
        process_language(df, token_map, lang, args)


def process_language(dfs, token_map, lang, args):
    for mode in ['train', 'val', 'test']:
        process_language_mode(dfs[mode], token_map, lang, mode, args)


def process_language_mode(df, token_map, lang, mode, args):
    data = parse_data(df, token_map, args)
    save_data(data, lang, mode, args.ffolder, fsuffix=args.fsuffix)


def parse_data(df, token_map, args):
    max_len = df.phon.map(lambda x: len(x)).max()
    data = np.zeros((df.shape[0], max_len + 3))

    for i, (index, x) in enumerate(df.iterrows()):
        try:
            instance = x.phon if not args.reverse else x.phon[::-1]
            data[i, 0] = 1
            data[i, 1:len(instance) + 1] = [token_map[z] for z in instance]
            data[i, len(instance) + 1] = 2
            data[i, -1] = x.new_id
        except:
            continue

    return data


def save_data(data, lang, mode, ffolder, fsuffix=''):
    with open('%s/preprocess%s/data-%s-%s.npy' % (ffolder, fsuffix, lang, mode), 'wb') as f:
        np.save(f, data)


def save_info(ffolder, languages, token_map, data_split, concepts_ids, IPA_to_concept, fsuffix=''):
    info = {
        'languages': languages,
        'token_map': token_map,
        'data_split': data_split,
        'concepts_ids': concepts_ids,
        'IPA_to_concept': IPA_to_concept,
    }
    with open('%s/preprocess%s/info.pckl' % (ffolder, fsuffix), 'wb') as f:
        pickle.dump(info, f)


def load_info(ffolder, fsuffix=''):
    with open('%s/preprocess%s/info.pckl' % (ffolder, fsuffix), 'rb') as f:
        info = pickle.load(f)
    languages = info['languages']
    token_map = info['token_map']
    data_split = info['data_split']
    concept_ids = info['concepts_ids']

    return languages, token_map, data_split, concept_ids


def main(args):
    languages = get_languages()
    df = read_src_data(args.ffolder, languages)

    word2vec = Word2VecInfo.get_word2vec(ffolder=args.ffolder)
    df = df[df.new_id.isin(word2vec)]

    train_df, val_df, test_df, data_split = separate_train(df)
    token_map = get_tokens(df)

    languages_df = separate_per_language(train_df, val_df, test_df, languages)

    process_languages(languages_df, token_map, args)
    save_info(args.ffolder, languages, token_map, data_split, None, None, fsuffix=args.fsuffix)


if __name__ == '__main__':
    args = parser.parse_args()
    assert args.data == 'celex', 'this script should only be run with northeuralex data'
    main(args)

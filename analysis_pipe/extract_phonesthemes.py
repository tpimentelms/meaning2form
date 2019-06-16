import pandas as pd
import numpy as np

import sys
sys.path.append('./')
from util import argparser
from util import util
from data_pipe.celex import CelexInfo


argparser.add_argument('--mi-type', type=str, default='mixed', choices=['word2vec', 'mixed'],
                       help='Should get word2vec mutual info, or mixed. Default: mixed')
argparser.add_argument('--n-permuts', type=int, default=100000,
                       help='Number of permutations in phonesthemes test. Default: 100000')


mi_pairs = {
    'word2vec': ('none', 'word2vec'),
    'mixed': ('pos', 'mixed'),
}
n_preffixes = [2, 3]


def get_full_df(model, args):
    df = pd.read_csv('%s/orig/lstm__%s__results-per-position-per-word.csv' % (args.rfolder, model))
    return df[['lang', 'phoneme_id', 'phoneme', 'phoneme_len']].set_index('phoneme_id')


def get_df(model, args):
    df = pd.read_csv('%s/orig/lstm__%s__results-per-position-per-word.csv' % (args.rfolder, model))
    for n_preffix in n_preffixes:
        df['%s__n-%d' % (model, n_preffix)] = df.apply(lambda x: sum([x['%d' % i] for i in range(n_preffix)]), axis=1)
    return df[['phoneme_id'] + ['%s__n-%d' % (model, i) for i in n_preffixes]].set_index('phoneme_id')


def get_data(models, args):
    df_final = get_full_df(models[0], args)
    for model in models:
        df = get_df(model, args)
        df_final = df_final.join(df)
    df_final['phoneme_list'] = df_final.phoneme.apply(lambda x: x.split(' '))
    return df_final.reset_index()


def get_phonestemes(df, lang, name, n=2, n_permuts=100000, is_suffix=False):
    if not is_suffix:
        df['preffix'] = df.phoneme_list.apply(lambda x: ''.join(x[:n]))
    else:
        df['preffix'] = df.phoneme_list.apply(lambda x: ''.join(x[n - 1::-1]))
    ngrams = set([x for x in df.preffix.unique() if len(x) >= n])
    ngrams_stats = {x: [] for x in ngrams}
    ngrams_idxs = {x: [] for x in ngrams}

    for idx, x in df.iterrows():
        if x.preffix not in ngrams:
            continue
        ngrams_stats[x.preffix] += [x.mutual_info]
        ngrams_idxs[x.preffix] += [x.phoneme_id]

    preffix_stats = [[
        preffix, np.mean(mutual_info), np.std(mutual_info, ddof=1), len(mutual_info), mutual_info]
        for preffix, mutual_info in ngrams_stats.items()]
    df_preffix = pd.DataFrame(preffix_stats, columns=['preffix', 'mi_avg', 'mi_std', 'occurences', 'info_list']) \
        .sort_values('mi_avg', ascending=False)

    df_celex = CelexInfo().df
    df_preffix['examples'] = df_preffix.preffix.apply(
        lambda x: list(df_celex.loc[ngrams_idxs[x][:10]].word.values) if len(x) > 1 else '-')

    words_info = [x for y in df_preffix.info_list for x in y]
    df_preffix['p_val'] = 5

    for i in range(6, df_preffix.occurences.max() + 1):
        df_temp = df_preffix[df_preffix.occurences == i].copy()
        if df_temp.shape[0] == 0:
            continue
        avgs = np.array([np.mean(np.random.choice(words_info, i)) for _ in range(n_permuts)])

        for idx, row in df_temp.iterrows():
            p_val = np.sum(row.mi_avg < avgs) / n_permuts
            df_preffix.loc[idx, 'p_val'] = p_val

    analysis_type = 'preffix' if not is_suffix else 'suffix'
    df_preffix.sort_values('p_val', ascending=True, inplace=True)
    del df_preffix['info_list']
    df_preffix.to_csv(
        'results/celex/phonesthemes/raw/preffix__%s__%s__%s__n-%d__permuts-%d.tsv' %
        (lang, name, analysis_type, n, n_permuts), sep='\t')


def main(args):
    print(args.rfolder)
    mi_pair = mi_pairs[args.mi_type]

    df = get_data(mi_pair, args)
    models = set([x[:-5] for x in df.columns
                  if x not in ['lang', 'avg_len', 'unigram', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_list']])

    if not set(mi_pair).issubset(models):
        print('Don\'t have an item in pair:', mi_pair)
        return

    for lang in df.lang.unique():
        df_lang = df[df.lang == lang].copy()

        for n in n_preffixes:
            print('Analysing pair: %s. Using %d-gram. Analysing language: %s.' % (mi_pair, n, lang))
            df_lang['mutual_info'] = df_lang['%s__n-%d' % (mi_pair[0], n)] - df_lang['%s__n-%d' % (mi_pair[1], n)]
            get_phonestemes(df_lang, lang, args.mi_type, n_permuts=args.n_permuts, n=n, is_suffix=args.suffix)

    print(models)


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv', orig_folder=False)
    args.suffix = args.reverse
    assert args.data == 'celex', 'this script should only be run with celex data'
    util.mkdir('results/celex/phonesthemes/raw/')
    main(args)

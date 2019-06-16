import pandas as pd
import argparse

import sys
sys.path.append('./')
from util import util


parser = argparse.ArgumentParser(description='Phoneme LM phonestheme analysis')
parser.add_argument('--mi-type', type=str, default='mixed',
                    help='Should get word2vec mutual info, or mixed. Default: mixed')
parser.add_argument('--n-permuts', type=int, default=100000,
                    help='Number of permutations in phonesthemes test. Default: 100000')


def get_dataframe(lang, n, is_preffix, args):
    mi_type = args.mi_type
    affix_str = 'preffix' if is_preffix else 'suffix'
    fname = 'results/celex/phonesthemes/raw/preffix__%s__%s__%s__n-%d__permuts-%d.tsv' \
        % (lang, mi_type, affix_str, n, args.n_permuts)
    df = pd.read_csv(
        fname, delimiter='\t', index_col=0)
    return df


def get_data(lang, is_preffix, args):
    dfs = [get_dataframe(lang, n, is_preffix, args) for n in [2, 3]]
    df = pd.concat(dfs)
    df.sort_values('p_val', ascending=True, inplace=True)
    return df


def get_preffix_data(lang, args):
    df = get_data(lang, is_preffix=True, args=args)
    df['-ffix'] = df.preffix.apply(lambda x: '%s-' % (x))
    return df


def get_suffix_data(lang, args):
    df = get_data(lang, is_preffix=False, args=args)
    df['-ffix'] = df.preffix.apply(lambda x: '-%s' % (x))
    return df


def get_all_data(lang, args):
    df_preffix = get_preffix_data(lang, args=args)
    df_suffix = get_suffix_data(lang, args=args)
    df = pd.concat([df_preffix, df_suffix], sort=False)
    df.sort_values('p_val', ascending=True, inplace=True)
    return df


def analyse_all(lang, args, min_occurrences=21, alphas=[.01, .05]):
    print('Analysing phonesthemes for language: %s' % lang)
    df = get_all_data(lang, args=args)

    df = df[df.occurences >= min_occurrences]
    df['rank'] = range(1, df.shape[0] + 1)
    for alpha in alphas:
        df['bh_threshold__%.2f' % alpha] = df['rank'].apply(lambda x: alpha * x / df.shape[0])
        df['valid__%.2f' % alpha] = df['p_val'] < df['bh_threshold__%.2f' % alpha]

    df.to_csv('results/celex/phonesthemes/full/phonesthemes_%s_%s.tsv' % (args.mi_type, lang), sep='\t')
    df_valid = df[df['valid__%.2f' % max(alphas)]]
    df_valid.to_csv('results/celex/phonesthemes/phonesthemes_%s_%s.tsv' % (args.mi_type, lang), sep='\t')


def main(args):
    for lang in ['eng', 'deu', 'nld']:
        analyse_all(lang, args)


if __name__ == '__main__':
    args = parser.parse_args()
    util.mkdir('results/celex/phonesthemes/full/')
    main(args)

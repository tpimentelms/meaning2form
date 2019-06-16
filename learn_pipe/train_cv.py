import numpy as np

import sys
sys.path.append('./')
from data_pipe.celex import CelexInfo
from data_pipe.northeuralex import NortheuralexInfo
from data_pipe.parse_northeuralex import load_info
from learn_pipe.model import opt_params
from learn_pipe.train import convert_to_loader, _run_language, write_csv
from util import argparser


full_results = [['lang', 'fold', 'avg_len', 'test_loss', 'test_acc', 'val_loss', 'val_acc', 'best_epoch']]


def get_data(args):
    if args.data == 'celex':
        dataset = CelexInfo()
    elif args.data == 'northeuralex':
        dataset = NortheuralexInfo()
    return dataset.df


def get_lang_df(lang, args):
    df = get_data(args)
    return df[df['Language_ID'] == lang]


def get_data_loaders_cv(fold, nfolds, lang, token_map, args, verbose=True):
    global data_split
    df = get_lang_df(lang, args)
    data_split = get_data_split_cv(fold, nfolds, df, args, verbose=verbose)

    train_loader = get_data_loader(df, data_split[0], token_map, 'train', args)
    val_loader = get_data_loader(df, data_split[1], token_map, 'val', args)
    test_loader = get_data_loader(df, data_split[2], token_map, 'test', args)

    return train_loader, val_loader, test_loader


def get_data_split_cv(fold, nfolds, df, args, verbose=True):
    _, _, data_split, _ = load_info(ffolder=args.ffolder)
    lang_concepts = set(df.map_id.unique())
    concepts = [y for x in data_split for y in x if y in lang_concepts]
    return _get_data_split_cv(fold, nfolds, concepts, verbose=verbose)


def _get_data_split_cv(fold, nfolds, concepts, verbose=True):
    part_size = int(len(concepts) / nfolds)
    test_fold = (fold + 1) % nfolds
    train_start_fold = 0 if test_fold > fold else (test_fold + 1)

    train = concepts[train_start_fold * part_size:fold * part_size]
    train += concepts[(fold + 2) * part_size:] if fold + 2 < nfolds else []
    val = concepts[fold * part_size:(fold + 1) * part_size] if fold + 1 < nfolds else concepts[fold * part_size:]
    test = concepts[(test_fold) * part_size:(test_fold + 1) * part_size] if test_fold + 1 < nfolds \
        else concepts[(test_fold) * part_size:]

    if verbose:
        print('Train %d, Val %d, Test %d' % (len(train), len(val), len(test)))

    return (train, val, test)


def get_data_loader(df, concepts, token_map, mode, args):
    data = split_data(df, concepts, token_map, mode, args)
    return convert_to_loader(data, mode)


def split_data(df, concepts, token_map, mode, args):
    df_partial = df[df['map_id'].isin(set(concepts))]
    data_partial = df_partial['phoneme'].values
    ids = df_partial.index

    max_len = max([len(x) for x in data_partial])
    data = np.zeros((len(data_partial), max_len + 3))
    data.fill(token_map['PAD'])
    for i, (string, _id) in enumerate(zip(data_partial, ids)):
        string = string if not args.reverse else string[::-1]
        _data = [token_map['SOW']] + [token_map[x] for x in string] + [token_map['EOW']]
        data[i, :len(_data)] = _data
        data[i, -1] = _id

    return data


def run_language_cv(
        lang, token_map, args, embedding_size=None, hidden_size=256, word2vec_size=10, nlayers=1, dropout=0.2):
    global full_results, fold
    nfolds = 10
    avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc = 0, 0, 0, 0
    for fold in range(nfolds):
        print()
        print('Fold:', fold, end=' ')

        train_loader, val_loader, test_loader = get_data_loaders_cv(fold, nfolds, lang, token_map, args)
        avg_len, test_loss, test_acc, \
            best_epoch, val_loss, val_acc = _run_language(
                lang, train_loader, val_loader, test_loader, token_map,
                args, embedding_size=embedding_size, hidden_size=hidden_size,
                word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)

        full_results += [[lang, fold, avg_len, test_loss, test_acc, val_loss, val_acc, best_epoch]]

        avg_test_loss += test_loss / nfolds
        avg_test_acc += test_acc / nfolds
        avg_val_loss += val_loss / nfolds
        avg_val_acc += val_acc / nfolds

        write_csv(full_results, '%s/%s__%s__full-results.csv' % (args.rfolder, args.model, args.context))

    return avg_len, avg_test_loss, avg_test_acc, avg_val_loss, avg_val_acc


def run_opt_language_cv(lang, token_map, args):
    embedding_size, hidden_size, word2vec_size, nlayers, dropout = opt_params.get_opt_params(lang, args)
    print('Optimum hyperparams emb-hs: %d, hs: %d, w2v: %d, nlayers: %d, drop: %.4f'
          % (embedding_size, hidden_size, word2vec_size, nlayers, dropout))

    return run_language_cv(lang, token_map, args,
                           embedding_size=embedding_size, hidden_size=hidden_size, word2vec_size=word2vec_size,
                           nlayers=nlayers, dropout=dropout)


def run_language_enveloper_cv(lang, token_map, args):
    if args.opt:
        return run_opt_language_cv(lang, token_map, args)
    else:
        return run_language_cv(lang, token_map, args)


def run_languages(args):
    languages, token_map, data_split, _ = load_info(ffolder=args.ffolder)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'avg_len', 'test_loss', 'test_acc', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        print()
        print('%d. Language %s' % (i, lang))

        avg_len, test_loss, test_acc, \
            val_loss, val_acc = run_language_enveloper_cv(lang, token_map, args)

        results += [[lang, avg_len, test_loss, test_acc, val_loss, val_acc]]

        write_csv(results, '%s/%s__%s__results.csv' % (args.rfolder, args.model, args.context))
    write_csv(results, '%s/%s__%s__results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='cv')
    run_languages(args)

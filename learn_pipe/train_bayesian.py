import numpy as np

from gp import bayesian_optimisation
from train import get_model_entropy, get_data_loaders, write_csv, run_language
from util import argparser
import sys
sys.path.append('./')
from data_pipe.parse_northeuralex import load_info


results = [['lang', 'embedding_size', 'hidden_size', 'word2vec_size', 'nlayers', 'dropout',
            'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
wait_epochs = 10


def sample_loss_getter(lang, token_map, args):
    global count
    train_loader, val_loader, test_loader = get_data_loaders(lang, args.ffolder, args)
    count = 0

    def sample_loss(hyper_params):
        global results, count
        count += 1

        embedding_size = int(hyper_params[0])
        hidden_size = int(hyper_params[1])
        word2vec_size = int(2 ** hyper_params[2])
        nlayers = int(max(1, hyper_params[3]))
        dropout = max(0, hyper_params[4])
        print('%d: emb-hs %d  hs %d  w2v %d  nlayers %d  drop %.3f' %
              (count, embedding_size, hidden_size, word2vec_size, nlayers, dropout))

        test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
            lang, train_loader, val_loader, test_loader, token_map, embedding_size, hidden_size,
            word2vec_size, nlayers, dropout, args, wait_epochs=wait_epochs, per_word=False)

        results += [[
            lang, embedding_size, hidden_size, word2vec_size, nlayers, dropout,
            test_loss, test_acc, best_epoch, val_loss, val_acc]]
        return val_loss

    return sample_loss


def get_optimal_loss(lang, token_map, xp, yp, args):
    best_hyperparams = xp[np.argmin(yp)]
    embedding_size = int(best_hyperparams[0])
    hidden_size = int(best_hyperparams[1])
    word2vec_size = int(2 ** best_hyperparams[2])
    nlayers = int(max(1, best_hyperparams[3]))
    dropout = max(0, best_hyperparams[4])
    print('Best hyperparams emb-hs: %d, hs: %d, w2v: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, word2vec_size, nlayers, dropout))

    avg_len, test_loss, test_acc, best_epoch, val_loss, val_acc = run_language(
        lang, token_map, args, embedding_size=embedding_size, hidden_size=hidden_size,
        word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)
    return [lang, avg_len, test_loss, test_acc, best_epoch, val_loss, val_acc,
            embedding_size, hidden_size, word2vec_size, nlayers, dropout]


def optimize_languages(args):
    languages, token_map, _, _ = load_info(ffolder=args.ffolder)
    print('Model %s' % args.model)

    n_iters = 45
    bounds = np.array([[4, 256], [32, 256], [.5, 8], [1, 2.95], [0.0, 0.5]])
    n_pre_samples = 5

    opt_results = [['lang', 'avg_len', 'test_loss', 'test_acc', 'best_epoch',
                    'val_loss', 'val_acc', 'embedding_size', 'hidden_size', 'word2vec_size',
                    'nlayers', 'dropout']]

    for i, lang in enumerate(languages):
        print()
        print('%d. %s' % (i, lang))
        sample_loss = sample_loss_getter(lang, token_map, args)
        xp, yp = bayesian_optimisation(n_iters, sample_loss, bounds, n_pre_samples=n_pre_samples)

        opt_results += [get_optimal_loss(lang, token_map, xp, yp, args)]

        write_csv(results, '%s/%s__%s__baysian-results.csv' % (args.rfolder, args.model, args.context))
        write_csv(opt_results, '%s/%s__%s__opt-results.csv' % (args.rfolder, args.model, args.context))

    write_csv(results, '%s/%s__%s__baysian-results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='bayes-opt')
    optimize_languages(args)

import pandas as pd


def _get_opt_params(fname, lang, delimiter='\t'):
    results = pd.read_csv(fname, delimiter=delimiter)
    instance = results[results['lang'] == lang]

    embedding_size = int(instance['embedding_size'].item())
    hidden_size = int(instance['hidden_size'].item())
    word2vec_size = int(instance['word2vec_size'].item())
    nlayers = int(instance['nlayers'].item())
    dropout = instance['dropout'].item()

    return embedding_size, hidden_size, word2vec_size, nlayers, dropout


def get_opt_params(lang, args):
    context = args.context if 'shuffle' not in args.context else args.context[:-8]
    fname = '%s/bayes-opt%s/orig/%s__%s__opt-results.csv' \
        % (args.rfolder_base, args.fsuffix, args.model, context)
    return _get_opt_params(fname, lang, delimiter=',')

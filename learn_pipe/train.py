import numpy as np
import math
import csv
from tqdm import tqdm
import io

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

import sys
sys.path.append('./')
from data_pipe.parse_northeuralex import load_info
from learn_pipe.model import opt_params
from learn_pipe.model.lstm import IpaLM
from util import argparser
from util import constants

results_per_word = [['lang', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_loss']]
results_per_position = [['lang'] + list(range(30))]
results_per_position_per_word = [['lang', 'phoneme_id', 'phoneme', 'phoneme_len', 'phoneme_loss'] + list(range(30))]


def get_data_loaders(lang, ffolder, args):
    train_loader = get_data_loader(lang, 'train', ffolder, args)
    val_loader = get_data_loader(lang, 'val', ffolder, args)
    test_loader = get_data_loader(lang, 'test', ffolder, args)

    return train_loader, val_loader, test_loader


def get_data_loader(lang, mode, ffolder, args):
    data = read_data(lang, mode, ffolder, args)
    return convert_to_loader(data, mode)


def read_data(lang, mode, ffolder, args):
    with open('%s/preprocess%s/data-%s-%s.npy' % (ffolder, args.fsuffix, lang, mode), 'rb') as f:
        data = np.load(f)

    return data


def write_csv(results, filename):
    with io.open(filename, 'w', encoding='utf8') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerows(results)


def convert_to_loader(data, mode, batch_size=64):
    x = torch.from_numpy(data[:, :-2]).long().to(device=constants.device)
    y = torch.from_numpy(data[:, 1:-1]).long().to(device=constants.device)
    idx = torch.from_numpy(data[:, -1]).long().to(device=constants.device)

    shuffle = True if mode == 'train' else False

    dataset = TensorDataset(x, y, idx)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def train_epoch(train_loader, model, loss, optimizer):
    model.train()
    total_loss = 0.0
    for batches, (batch_x, batch_y, batch_idx) in enumerate(train_loader):
        optimizer.zero_grad()
        y_hat, _ = model(batch_x, batch_idx)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        l.backward()
        optimizer.step()

        total_loss += l.item()
    return total_loss / (batches + 1)


def eval(data_loader, model, loss):
    model.eval()
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    for batches, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
        y_hat, _ = model(batch_x, batch_idx)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)) / math.log(2)
        val_loss += l.item() * batch_y.size(0)

        non_pad = batch_y != 0
        val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    return val_loss, val_acc


def run_model(model, batch_x, batch_idx):
    return model(batch_x, batch_idx)


def eval_per_word(lang, data_loader, model, token_map, args, model_func=run_model):
    global results_per_word, results_per_position, results_per_position_per_word
    model.eval()

    token_map_inv = {x: k for k, x in token_map.items()}
    ignored_tokens = [token_map['PAD'], token_map['SOW'], token_map['EOW']]
    loss = nn.CrossEntropyLoss(ignore_index=0, reduction='none').to(device=constants.device)
    val_loss, val_acc, total_sent = 0.0, 0.0, 0
    loss_per_position, count_per_position = None, None

    for batches, (batch_x, batch_y, batch_idx) in enumerate(data_loader):
        y_hat, _ = model_func(model, batch_x, batch_idx)
        l = loss(y_hat.view(-1, y_hat.size(-1)), batch_y.view(-1)).reshape_as(batch_y).detach() / math.log(2)

        loss_per_position = loss_per_position + l.sum(0).data if loss_per_position is not None else l.sum(0).data
        count_per_position = count_per_position + (l != 0).sum(0).data if count_per_position is not None else (l != 0).sum(0).data

        words = torch.cat([batch_x, batch_y[:, -1:]], -1).detach()

        words_ent = l.sum(-1)
        words_len = (batch_y != 0).sum(-1)

        words_ent_avg = words_ent / words_len.float()
        val_loss += words_ent_avg.sum().item()

        non_pad = batch_y != 0
        val_acc += (y_hat.argmax(-1)[non_pad] == batch_y[non_pad]).float().mean().item() * batch_y.size(0)

        total_sent += batch_y.size(0)

        for i, w in enumerate(words):
            # _w = [token_map_inv[x] for x in w.tolist() if x not in ignored_tokens]
            _w = idx_to_word(w, token_map_inv, ignored_tokens)
            idx = batch_idx[i].item()
            results_per_word += [[lang, idx, _w, words_len[i].item(), words_ent_avg[i].item()]]
            results_per_position_per_word += [[lang, idx, _w, words_len[i].item(), words_ent_avg[i].item()] +
                                              l[i].float().cpu().numpy().tolist()]

    results_per_position += [[lang] + list((loss_per_position / count_per_position.float()).cpu().numpy())]
    val_loss = val_loss / total_sent
    val_acc = val_acc / total_sent

    write_csv(results_per_word, '%s/%s__%s__results-per-word.csv' % (args.rfolder, args.model, args.context))
    write_csv(results_per_position, '%s/%s__%s__results-per-position.csv' % (args.rfolder, args.model, args.context))
    write_csv(results_per_position_per_word,
              '%s/%s__%s__results-per-position-per-word.csv' % (args.rfolder, args.model, args.context))

    return val_loss, val_acc, results_per_word


def word_to_tensors(word, token_map):
    w = word_to_idx(word, token_map)

    x = torch.from_numpy(w[:, :-1]).long().to(device=constants.device)
    y = torch.from_numpy(w[:, 1:]).long().to(device=constants.device)
    return x, y


def word_to_idx(word, token_map):
    w = [[token_map['SOW']] + [token_map[x] for x in word] + [token_map['EOW']]]
    return np.array(w)


def idx_to_word(word, token_map_inv, ignored_tokens):
    _w = [token_map_inv[x] for x in word.tolist() if x not in ignored_tokens]
    return ' '.join(_w)


def train(train_loader, val_loader, test_loader, model, loss, optimizer, wait_epochs=50):
    epoch, best_epoch, best_loss, best_acc = 0, 0, float('inf'), 0.0

    pbar = tqdm(total=wait_epochs)
    while True:
        epoch += 1

        total_loss = train_epoch(train_loader, model, loss, optimizer)
        val_loss, val_acc = eval(val_loader, model, loss)

        if val_loss < best_loss:
            best_epoch = epoch
            best_loss = val_loss
            best_acc = val_acc
            model.set_best()

        pbar.total = best_epoch + wait_epochs
        pbar.update(1)
        pbar.set_description(
            '%d/%d: train_loss %.3f  val_loss: %.3f  acc: %.3f  best_loss: %.3f  acc: %.3f' %
            (epoch, best_epoch, total_loss, val_loss, val_acc, best_loss, best_acc))

        if epoch - best_epoch >= wait_epochs:
            break

    pbar.close()
    model.recover_best()

    return best_epoch, best_loss, best_acc


def _get_avg_len(data_loader):
    total_phon, total_sent = 0.0, 0.0
    for batches, (batch_x, batch_y, _) in enumerate(data_loader):
        batch = torch.cat([batch_x, batch_y[:, -1:]], dim=-1)
        total_phon += (batch != 0).sum().item()
        total_sent += batch.size(0)

    avg_len = (total_phon * 1.0 / total_sent) - 2  # Remove SOW and EOW tag in every sentence

    return avg_len, total_sent


def get_avg_len(data_loaders):
    total_len, total_nsent = 0, 0
    for data_loader in data_loaders:
        length, nsentences = _get_avg_len(data_loader)
        total_len += (length * nsentences)
        total_nsent += nsentences

    return total_len * 1.0 / total_nsent


def init_model(model_name, context, hidden_size, word2vec_size, token_map, embedding_size, nlayers, dropout, args):
    vocab_size = len(token_map)
    if model_name == 'lstm':
        model = IpaLM(
            vocab_size, hidden_size, embedding_size=embedding_size, word2vec_size=word2vec_size,
            nlayers=nlayers, dropout=dropout, context=context, data=args.data).to(device=constants.device)
    else:
        raise ValueError("Model not implemented: %s" % model_name)

    return model


def get_model_entropy(
        lang, train_loader, val_loader, test_loader, token_map, embedding_size, hidden_size, word2vec_size,
        nlayers, dropout, args, wait_epochs=50, per_word=True):
    model = init_model(
        args.model, args.context, hidden_size, word2vec_size, token_map,
        embedding_size, nlayers, dropout, args)

    loss = nn.CrossEntropyLoss(ignore_index=0).to(device=constants.device)
    optimizer = optim.Adam(model.parameters())

    best_epoch, val_loss, val_acc = train(train_loader, val_loader, test_loader, model,
                                          loss, optimizer, wait_epochs=wait_epochs)

    if per_word:
        test_loss, test_acc, _ = eval_per_word(lang, test_loader, model, token_map, args)
    else:
        test_loss, test_acc = eval(test_loader, model, loss)
    model.save(args.cfolder, args.context, lang)

    return test_loss, test_acc, best_epoch, val_loss, val_acc


def _run_language(
        lang, train_loader, val_loader, test_loader, token_map, args, embedding_size=None,
        hidden_size=256, word2vec_size=10, nlayers=1, dropout=0.2, per_word=True):
    avg_len = get_avg_len([train_loader, val_loader, test_loader])
    print('Language %s Avg len: %.4f' % (lang, avg_len))

    test_loss, test_acc, best_epoch, val_loss, val_acc = get_model_entropy(
        lang, train_loader, val_loader, test_loader, token_map, embedding_size, hidden_size,
        word2vec_size, nlayers, dropout, args, per_word=per_word)
    print('Test loss: %.4f  acc: %.4f    Avg len: %.4f' % (test_loss, test_acc, avg_len))

    return avg_len, test_loss, test_acc, best_epoch, val_loss, val_acc


def run_language(lang, token_map, args, embedding_size=None, hidden_size=256, word2vec_size=10, nlayers=1, dropout=0.2):
    train_loader, val_loader, test_loader = get_data_loaders(lang, ffolder=args.ffolder, args=args)

    return _run_language(lang, train_loader, val_loader, test_loader, token_map,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)


def run_opt_language(lang, token_map, args):
    train_loader, val_loader, test_loader = get_data_loaders(lang, ffolder=args.ffolder, args=args)
    embedding_size, hidden_size, word2vec_size, nlayers, dropout = opt_params.get_opt_params(lang, args)
    print('Optimum hyperparams emb-hs: %d, hs: %d, nlayers: %d, drop: %.4f' %
          (embedding_size, hidden_size, nlayers, dropout))

    return _run_language(lang, train_loader, val_loader, test_loader, token_map,
                         args, embedding_size=embedding_size, hidden_size=hidden_size,
                         word2vec_size=word2vec_size, nlayers=nlayers, dropout=dropout)


def run_language_enveloper(lang, token_map, args):
    if args.opt:
        return run_opt_language(lang, token_map, args)
    else:
        return run_language(lang, token_map, args)


def run_languages(args):
    languages, token_map, data_split, _ = load_info(ffolder=args.ffolder)
    print('Train %d, Val %d, Test %d' % (len(data_split[0]), len(data_split[1]), len(data_split[2])))

    results = [['lang', 'avg_len', 'test_loss', 'test_acc', 'best_epoch', 'val_loss', 'val_acc']]
    for i, lang in enumerate(languages):
        print()
        print(i, end=' ')

        avg_len, test_loss, test_acc, \
            best_epoch, val_loss, val_acc = run_language_enveloper(lang, token_map, args)

        results += [[lang, avg_len, test_loss, test_acc, best_epoch, val_loss, val_acc]]

        write_csv(results, '%s/%s__%s__results.csv' % (args.rfolder, args.model, args.context))
    write_csv(results, '%s/%s__%s__results-final.csv' % (args.rfolder, args.model, args.context))


if __name__ == '__main__':
    args = argparser.parse_args(csv_folder='normal')
    run_languages(args)

import argparse
from . import util

parser = argparse.ArgumentParser(description='Phoneme LM')

# Data
parser.add_argument('--data', type=str, default='northeuralex',
                    help='Dataset used. (default: northeuralex)')
parser.add_argument('--data-path', type=str, default='datasets',
                    help='Path where data is stored.')
parser.add_argument('--reverse', action='store_true', default=False,
                    help='Should use reversed strings in experiment.')

# Model
parser.add_argument('--model', default='lstm', choices=['lstm'],
                    help='Model used. (default: lstm)')
parser.add_argument('--context', default='none', choices=['none', 'pos', 'word2vec', 'mixed'],
                    help='Context used for systematicity. (default: none)')
parser.add_argument('--opt', action='store_true', default=False,
                    help='Should use optimum parameters in training.')

# Others
parser.add_argument('--results-path', type=str, default='results',
                    help='Path where results should be stored.')
parser.add_argument('--checkpoint-path', type=str, default='checkpoints',
                    help='Path where checkpoints should be stored.')
parser.add_argument('--csv-folder', type=str, default=None,
                    help='Specific path where to save results.')
parser.add_argument('--seed', type=int, default=7,
                    help='Seed for random algorithms repeatability (default: 7)')


def add_argument(*args, **kwargs):
    return parser.add_argument(*args, **kwargs)


def set_defaults(*args, **kwargs):
    return parser.set_defaults(*args, **kwargs)


def get_default(*args, **kwargs):
    return parser.get_default(*args, **kwargs)


def parse_args(*args, csv_folder='', orig_folder=True, **kwargs):
    args = parser.parse_args(*args, **kwargs)
    csv_folder = csv_folder if csv_folder != 'normal' or not args.opt else 'opt'
    csv_folder = csv_folder if args.csv_folder is None else args.csv_folder
    args.ffolder = '%s/%s' % (args.data_path, args.data)  # Data folder
    args.fsuffix = '' if not args.reverse else '_inv'

    args.rfolder_base = '%s/%s' % (args.results_path, args.data)  # Results base folder

    if orig_folder:
        args.rfolder = '%s/%s%s/orig' % (args.rfolder_base, csv_folder, args.fsuffix)  # Results folder
    else:
        args.rfolder = '%s/%s%s' % (args.rfolder_base, csv_folder, args.fsuffix)  # Results folder

    args.cfolder = '%s/%s/%s%s' % (args.checkpoint_path, args.data, csv_folder, args.fsuffix)  # Checkpoint folder
    util.mkdir(args.rfolder)
    util.mkdir(args.cfolder)
    util.config(args.seed)
    return args

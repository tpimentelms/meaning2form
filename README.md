# meaning2form
Studying the arbitrariness of the sign through a systematicity analysis.

## Install Dependencies

Create a conda environment with
```bash
$ source config/conda.sh
```
And install your appropriate version of [PyTorch](https://pytorch.org/get-started/locally/).

## Parse data

```bash
$ python data_pipe/parse_celex.py --data celex
$ python data_pipe/parse_celex.py --data celex --reverse
$ python data_pipe/parse_northeuralex.py --data northeuralex
```

## Train models

```bash
$ python learn_pipe/train.py --data <data> --context <context>
```

Or train all at once:

```bash
$ python learn_pipe/train_multi.sh
```

Context can be:
* none: No context used
* word2vec: Word2Vec context used
* pos: Grammar class context used
* mixed: Grammar class and Word2vec contexts used


## Get Phonesthemes

Extract possible phonesthemes and test them by running:
```
$ python analysis_pipe/extract_phonesthemes.py --data celex --n-permuts 100000
$ python analysis_pipe/extract_phonesthemes.py --data celex --n-permuts 100000 --reverse
$ python analysis_pipe/analyse_phonesthemes.py --n-permuts 100000
```

## Extra Information

This project was tested with libraries:
```bash
numpy==1.11.3
pandas==0.23.4
scikit-learn==0.21.2
gensim==3.7.3
matplotlib==2.0.2
seaborn==0.9.0
tqdm==4.32.1
torch==1.1.0
```

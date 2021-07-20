import warnings
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from preprocessing import (
    tokenize_texts,
    merge_texts,
    search_non_ascii
)

FIG_SIZE = (10, 8)
FONT_SIZE = 14

def set_temp_matplotlib_params():
    """
    Sets temporary matplotlib params
    """
    plt.rcParams.update({'font.size': FONT_SIZE})


def plot_tags_distribution(tags, labels=True, title='Distribution of tags in text'):
    """
    Plot tags distribution

    :param tags: list or numpy array
        Tags to visualize
    :param labels: bool, optional, default = True
        Put labels on top of bars
    :param: title, str, optional, default = 'Distribution of tags in text'
        Plots title
    :return None
        Plots tags distribution

    :author Marcin Rybiński
    """
    tags_distribution = Counter(tags)
    x = np.array(list(tags_distribution.keys()))
    y_counts = np.array(list(tags_distribution.values()))
    y_dist = y_counts / y_counts.sum()

    set_temp_matplotlib_params()
    plt.figure(1, FIG_SIZE)
    plt.bar(x, y_counts)
    if labels:
        for idx, _ in enumerate(x):
            value_count = y_counts[idx]
            value_dist = y_dist[idx]
            value_text = f'{value_dist * 100:.2f}%'
            plt.text(idx, value_count, value_text)

    plt.ylim([0, 1.1 * y_counts.max()])
    plt.ylabel('Counts')
    plt.xlabel('Type of tag')
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_text_distribution(texts, n_top_words, labels=False,
                           xlabel='Words', title='Distribution of words'):
    """
    Plot texts distribution

    :param texts: list or numpy array
        Tags to visualize
    :param n_top_words: int
        Number of top words to show
    :param labels: bool, optional, default = True
        Put labels on top of bars
    :param: xlabel, str, optional, default = 'Words'
        Label on x axis
    :param: title, str, optional, default = 'Distribution of words'
        Plots title
    :return None
        Plots texts distribution

    :author Marcin Rybiński
    """
    texts_tokenized = tokenize_texts(texts)
    texts_merged = merge_texts(texts_tokenized)
    texts_distribution = Counter(texts_merged).most_common(n_top_words)
    texts_distribution = np.array(texts_distribution)

    x = texts_distribution[:, 0]
    y_counts = texts_distribution[:, 1].astype('int')
    y_dist = y_counts / y_counts.sum()

    set_temp_matplotlib_params()
    plt.figure(1, FIG_SIZE)
    plt.bar(x, y_counts)
    if labels:
        for idx, _ in enumerate(x):
            value_count = y_counts[idx]
            value_dist = y_dist[idx]
            value_text = f'{value_dist * 100:.2f}%'
            plt.text(idx - 0.3, value_count + 100, value_text)

    plt.ylim([0, 1.1 * y_counts.max()])
    plt.ylabel('Counts')
    plt.xlabel(xlabel)
    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.show()


def plot_non_ascii_distribution(texts, **kwargs):
    """
    Plot non ascii characters distribution

    :param texts: list or numpy array
        Non ascii characters to visualize
    :param **kwargs:
        **kwargs passed to plot_text_distribution()
    :return None
        Plots texts distribution

    :author Marcin Rybiński
    """
    texts_non_ascii = search_non_ascii(texts)
    texts_non_ascii = [text for text in texts_non_ascii if text != '']
    texts_non_ascii = tokenize_texts(texts_non_ascii, split_point='')
    texts_non_ascii = merge_texts(texts_non_ascii)

    # matplotlib cant render all of non ascii characters
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plot_text_distribution(texts_non_ascii, **kwargs)


def calc_top_words_distribution(texts, tags, n_top_words=10, **kwargs):
    """
    Calculate top words in each tag

    :param texts: list or numpy array
        Texts
    :param tags: list or numpy array
        Tags
    :param n_top_words: int, optional, default = 10
        Numer of top words
    :param **kwargs:
        **kwargs passed to texts_tokenized()
    :return pandas DatFrame
        Top words distribution

    :author Marcin Rybiński
    """
    tags_top_words_distribution = {}
    all_top_words = set()

    for tag in np.unique(tags):
        idx_tags = tags == tag
        texts_tokenized = tokenize_texts(texts[idx_tags], **kwargs)
        texts_merged = merge_texts(texts_tokenized)
        texts_distribution = Counter(texts_merged).most_common(n_top_words)
        texts_distribution = np.array(texts_distribution)
        texts_distribution[:, 1] = np.arange(1, texts_distribution.shape[0] + 1)
        texts_distribution = pd.DataFrame(
            data=texts_distribution[:, 1].astype('int'),
            index=texts_distribution[:, 0],
            columns=[tag]
        )
        tags_top_words_distribution[tag] = texts_distribution
        all_top_words |= set(texts_distribution.index)

    top_words_distribution = pd.DataFrame(index=all_top_words)
    for tag, distribution in tags_top_words_distribution.items():
        top_words_distribution = (
            top_words_distribution
            .join(distribution, how='left')
        )

    top_words_distribution['rank'] = np.mean(top_words_distribution.values, axis=1)
    top_words_distribution = top_words_distribution.sort_values(by='rank')

    return top_words_distribution

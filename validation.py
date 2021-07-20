import time

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    f1_score
)


def show_top_n_words_naive_bayes(classifier, vectorizer, n_top_words=10):
    """
    Show top words in each tag for Naive Bayes

    :param classifier: Naive Bayes
        Fitted Naive Bayes classifier
    :param vectorizer: Vectorizer
        Fitted Vectorizer
    :param n_top_words: int, optional, default = 10
        Numer of top words
    :return dict
        Top words distribution

    :author Marcin Rybiński
    """
    top_words = {}
    for idx, tag in enumerate(classifier.classes_):
        words = vectorizer.get_feature_names()
        words_with_log_prob = list(zip(words, classifier.feature_log_prob_[idx]))
        words_with_log_prob_sorted = sorted(words_with_log_prob, key=lambda t: t[1], reverse=True)
        top_words[tag] = np.array(words_with_log_prob_sorted[:n_top_words])[:, 0]
    return top_words


def show_confusion_matrix(y_true, y_pred, title='Confusion matrix'):
    """
    Plots confusion matrix

    :param y_true: list or numpy array
        True tags labels
    :param y_pred: list or numpy array
        Predicted tags labels
    :param title: str, optional, default = 'Confusion matrix'
        Plot's title
    :return None
        Plots confusion matrix

    :author Marcin Rybiński
    """
    mat = confusion_matrix(y_true, y_pred)
    mat = mat/mat.sum(axis=1)[:, None]
    mat = np.round(mat, 4)
    plt.rcParams.update({'font.size': 14})
    figure = plt.figure(1, (10, 10))
    figure.subplots_adjust(left=0.2)
    labels = sorted(np.unique(y_true))
    sns.heatmap(mat.T, square=True, annot=True, fmt='.2%', cbar=False,
                xticklabels=labels,
                yticklabels=labels)
    plt.yticks(rotation=0)
    plt.xlabel('Real categories')
    plt.ylabel('Predicted categories')
    plt.title(title)
    plt.show()


def show_classification_report(y_true, y_pred, **kwargs):
    """
    1) Calculates and prints precision, recall, f1 score, accuracy and f1 micro-average
    2) Plots confusion matrix

    :param y_true: list or numpy array
        True tags labels
    :param y_pred: list or numpy array
        Predicted tags labels
    :param **kwargs:
        kwargs passed to show_confusion_matrix()
    :return None
        Prints metrics and plots confusion matrix

    :author Marcin Rybiński
    """
    show_confusion_matrix(y_true, y_pred, **kwargs)
    print(classification_report(y_true, y_pred, zero_division=0))
    f1_micro = f1_score(y_true, y_pred, average='micro')
    print(f'F1 micro-average {f1_micro}')


def check_inference_time(texts, cleaning_pipeline, classifier_pipeline):
    """
    Calculates mean and std inference time

    Iterates over texts:
        1. apply cleaning pipeline
        2. predict with classifier pipeline
    for each text separetly.

    :param texts: list or numpy array
        True tags labels
    :param cleaning_pipeline: sklearn Pipeline
        Cleaning pipeline
    :param classifier_pipeline: sklearn Pipeline
        Classifier pipeline
    :return None
        Prints mean and std inference time

    :author Marcin Rybiński
    """
    check_is_fitted(classifier_pipeline.steps[-1][1])
    all_steps = cleaning_pipeline.steps.copy()
    all_steps += classifier_pipeline.steps.copy()
    full_pipe = Pipeline(all_steps)

    times = []
    for text in texts:
        try:
            time_start = time.time()
            full_pipe.predict(text)
            time_end = time.time() - time_start
            times.append(time_end)
        # gdy pusty text, bez tokenow
        except ValueError:
            continue

    mean_inference_time = np.mean(times)
    std_inference_time = np.std(times)
    print(f'Inference time: {mean_inference_time:.6f} +- {2*std_inference_time:.6f} seconds')

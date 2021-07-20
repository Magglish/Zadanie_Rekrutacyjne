import os
import random
import sys
import time

import pickle

import numpy as np

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC

from preprocessing import (
    read_texts_and_tags,
    WordCaser,
    PolishLetterReplacer,
    PunctuationRemover,
    StopWordsRemover,
    CurrencyRemover,
    UnitsRemover,
    NumberReplacer,
    WhitespaceRemover,
    NonAsciiCharacterWhitespaceSurrounder
)

import mlflow
import logging

logging.basicConfig(filename='log.txt', filemode='w')
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

if __name__ == '__main__':

    path_data_folder = os.path.join(os.getcwd(), 'data')

    logger.info('Collecting new data')
    try:
        texts_train, tags_train = read_texts_and_tags(path_data_folder, 'train')
        texts_test, tags_test = read_texts_and_tags(path_data_folder, 'test')
    except Exception as e:
        logger.exception(f'Unable to read data. Error: {e}')
    logger.info('New data collected')

    # in best scenario, it should be passed from CLI
    # ---- parser = argparse.ArgumentParser() ----
    # but in this example, we skip it
    run_name = 'Support Vector Classifier v01'
    case = 'lower'
    search_non_ascii_pattern = r'[^\x00-\x7F]'
    additional_stop_words = ['anonymized_account', 'rt']
    additional_currency = None
    additional_units = None
    additional_punctuation = ['\\\\n', '\\.', '/', '%', '\\\\', '@', '😁', '😂']
    number_action = 'delete'
    number_replacement_str = 'NUMBER'
    seed = 2021

    logger.info('Data verification and validation')
    # ---- Data verification and validation code ----

    with mlflow.start_run(run_name=run_name):
        start_time = time.time()
        logger.info(f'Starting MLFlow run: {run_name}')
        np.random.seed(seed)
        random.seed(seed)

        with open('texts_train.pickle', 'wb') as file:
            pickle.dump(texts_train, file)

        with open('texts_test.pickle', 'wb') as file:
            pickle.dump(texts_test, file)

        with open('tags_train.pickle', 'wb') as file:
            pickle.dump(tags_train, file)

        with open('tags_test.pickle', 'wb') as file:
            pickle.dump(tags_test, file)

        mlflow.log_artifact('texts_train.pickle')
        mlflow.log_artifact('texts_test.pickle')
        mlflow.log_artifact('tags_train.pickle')
        mlflow.log_artifact('tags_test.pickle')

        logger.info('Creating validation set')
        indices = np.arange(texts_train.shape[0])
        texts_train, texts_val, tags_train, tags_val, idx_train, idx_val = train_test_split(
            texts_train,
            tags_train,
            indices,
            test_size=0.3,
            random_state=seed,
            stratify=tags_train
        )
        with open('idx_train.pickle', 'wb') as file:
            pickle.dump(idx_train, file)

        with open('idx_val.pickle', 'wb') as file:
            pickle.dump(idx_val, file)

        mlflow.log_artifact('idx_train.pickle')
        mlflow.log_artifact('idx_val.pickle')

        logger.info('Creating cleaning pipeline')
        pipeline_steps = [
            ('word_caser', WordCaser(case)),
            ('polish_letter_replacer', PolishLetterReplacer()),
            ('non_ascii_character_whitespace_sorrounder', NonAsciiCharacterWhitespaceSurrounder(search_non_ascii_pattern)),
            ('whitespace_remover_1', WhitespaceRemover()),
            ('stop_words_remover', StopWordsRemover(additional_stop_words)),
            ('punctuation_remover', PunctuationRemover(additional_punctuation)),
            ('currency_remover', CurrencyRemover(additional_currency)),
            ('units_remover', UnitsRemover(additional_units)),
            ('number_replacer', NumberReplacer(number_action, number_replacement_str)),
            ('whitespace_remover_2', WhitespaceRemover())
        ]
        cleaning_pipeline = Pipeline(pipeline_steps)

        logger.info('Fitting cleaning pipeline and transforming training set')
        texts_train_cleaned = cleaning_pipeline.fit_transform(texts_train)

        logger.info('Transforming validation set')
        texts_val_cleaned = cleaning_pipeline.transform(texts_val)

        logger.info('Performing hyperparatemer optimization')
        # ---- Hyperparameter optimization code ----

        tfidf_params = {
            'encoding': 'utf-8',
            'decode_error': 'strict',
            'strip_accents': None,
            'lowercase': False,
            'preprocessor': None,
            'tokenizer': None,
            'analyzer': 'word',
            'stop_words': None,
            'token_pattern': r'(?u)\b\w\w\w+\b|[^\x00-\x7F]',
            'ngram_range': (1, 1),
            'max_df': 0.95,
            'min_df': 5
        }
        mlflow.log_params(tfidf_params)

        svc_params = {
            'C': 1.0,
            'kernel': 'poly',
            'degree': 3
        }
        mlflow.log_params(svc_params)

        logger.info('Hyperparatemer optimization completed. '
                    f'Best params are: '
                    f'tfidf: {tfidf_params} '
                    f'ml model: {svc_params}')

        logger.info('Creating ML model pipeline')
        svc_pipeline = Pipeline([
            ('tfidf_vectorizer', TfidfVectorizer(**tfidf_params)),
            ('svc', SVC(**svc_params)),
        ])

        logger.info('Fitting ML model')
        svc_pipeline.fit(texts_train_cleaned, tags_train)

        logger.info('Performing check on training data')
        logger.info('Predicting training set')
        tags_train_pred = svc_pipeline.predict(texts_train_cleaned)

        logger.info('Evaluating training set')
        f1_micro_train = f1_score(tags_train, tags_train_pred, average='micro')
        mlflow.log_metric('f1_micro_train_set', f1_micro_train)

        logger.info('Performing check on validation data')
        logger.info('Predicting validation set')
        tags_val_pred = svc_pipeline.predict(texts_val_cleaned)

        logger.info('Evaluating validation set')
        f1_micro_val = f1_score(tags_val, tags_val_pred, average='micro')
        mlflow.log_metric('f1_micro_validation_set', f1_micro_val)

        logger.info(f'F1 micro average -> '
                    f'train set = {f1_micro_train:.4f}, '
                    f'validation set = {f1_micro_val:.4f}')

        logger.info('Performing check on test data')
        tags_train = np.append(
            tags_train, tags_val
        )
        texts_train = np.append(
            texts_train, texts_val
        )
        logger.info('Fitting cleaning pipeline and transforming training set on full data')
        texts_train_cleaned = cleaning_pipeline.fit_transform(texts_train)

        with open('cleaning_pipeline.pickle', 'wb') as file:
            pickle.dump(cleaning_pipeline, file)
        mlflow.log_artifact('cleaning_pipeline.pickle')

        logger.info('Transforming test set')
        texts_test_cleaned = cleaning_pipeline.transform(texts_test)

        logger.info('Fitting ML model on full data')
        svc_pipeline.fit(texts_train_cleaned, tags_train)

        with open('model.pickle', 'wb') as file:
            pickle.dump(svc_pipeline, file)
        mlflow.log_artifact('model.pickle')

        logger.info('Predicting test set')
        tags_test_pred = svc_pipeline.predict(texts_test_cleaned)

        logger.info('Evaluating test set')
        f1_micro_test = f1_score(tags_test, tags_test_pred, average='micro')
        mlflow.log_metric('f1_micro_test_set', f1_micro_test)

        logger.info('Checking new ML model vs. production ML model')
        production_model_f1_metrics_path = os.path.join(
            os.getcwd(), 'production', 'metrics', 'f1_micro_test_set'
        )
        with open(production_model_f1_metrics_path, 'r') as file:
            production_model_f1 = file.readlines()[0].split(' ')[1]
            production_model_f1 = float(production_model_f1)

        f1_diff = f1_micro_test - production_model_f1
        logger.info(f'F1 micro average -> '
                    f'New model = {f1_micro_test:.4f}, '
                    f'Production = {production_model_f1:.4f}, '
                    f'Baseline = 0.8660')

        mlflow.log_artifact('model.pickle')

        end_time = time.time() - start_time
        logger.info(f'Ended MLFlow run: {run_name} after '
                    f'{end_time:.2f} seconds = '
                    f'{end_time/60:.2f} minutes = '
                    f'{end_time/3600:.2f} hours')

        mlflow.log_artifact('log.txt')

        mlflow.log_param('additional_punctuation', additional_punctuation)
        mlflow.log_param('case', case)
        mlflow.log_param('search_non_ascii_pattern', search_non_ascii_pattern)
        mlflow.log_param('additional_stop_words', additional_stop_words)
        mlflow.log_param('additional_currency', additional_currency)
        mlflow.log_param('additional_units', additional_units)
        mlflow.log_param('number_action', number_action)
        mlflow.log_param('number_replacement_str', number_replacement_str)
        mlflow.log_param('additional_punctuation', additional_punctuation)
        mlflow.log_param('seed', seed)

    mlflow.end_run()

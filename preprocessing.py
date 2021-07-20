import os
import re

import numpy as np
from copy import deepcopy

# import transformers as tfr
# import torch

from spacy.lang.pl.stop_words import STOP_WORDS
from spacy.lang.pl.punctuation import (
    PUNCT, UNITS, CURRENCY, LIST_QUOTES, LIST_HYPHENS, LIST_ELLIPSES
)

from sklearn.base import BaseEstimator, TransformerMixin


def read_txt(path):
    """
    Reads txt file

    :param path: str
        Path to the txt file
    :return: list
        Readed txt file

    :author Marcin Rybiński
    """
    texts = []
    with open(path) as file:
        for line in file:
            # txt files are always readed with enter ('\n')
            # at the end of the line
            # we are getting rid of it, but making sure
            # that all the other enters ale left unchanged
            line = line[:-1] if line.endswith('\n') else line
            texts.append(line)
    return texts


def read_texts_and_tags(folder_path, data_type):
    """
    Reads data for PolEval 2019 6.2 task

    This function assumes that the names of data files are exactly the same
    like you would download them from PolEval site

    :param folder_path: str
        Path to folder
    :param data_type: str
        What data we want to read - 'train' or 'test'
    :return: tuple(numpy array, numpy array)
        Tuple of readed text and tags in numpy array

    :author Marcin Rybiński
    """
    file_names = {
        'train': {
            'text': 'training_set_clean_only_text.txt',
            'tags': 'training_set_clean_only_tags.txt'
        },
        'test': {
            'text': 'test_set_only_text.txt',
            'tags': 'test_set_only_tags.txt'
        }
    }

    # i'm changing tags name for easier interpretation
    tags_meanings = {
        '0': 'non-harmful',
        '1': 'cyberbullying',
        '2': 'hate-speech'
    }

    texts_file_path = os.path.join(folder_path, file_names[data_type]['text'])
    tags_file_path = os.path.join(folder_path, file_names[data_type]['tags'])

    texts = read_txt(texts_file_path)
    tags = read_txt(tags_file_path)
    tags = [tags_meanings[tag] for tag in tags]

    return np.array(texts), np.array(tags)


def tokenize_texts(texts, split_point=' '):
    """
    Tokenize text

    :param texts: list
        Texts in list
    :param split_point: str, optional, default = ' '
        Split point for which we divide text into token
    :return: list
        Tokenized text

    :author Marcin Rybiński
    """
    if split_point == '':
        return np.array([list(text) for text in texts], dtype=object)
    else:
        return np.array([text.split() for text in texts], dtype=object)


def merge_texts(texts):
    """
    Merge texts into one text

    :param texts: list
        List of texts to be merged
    :return: numpy array
        Merged texts into one

    :author Marcin Rybiński
    """
    return np.array([word for text in texts for word in text])


def search_non_ascii(texts):
    """
    Texts in which we are looking for non ascii characters

    This function assumes that you are intrested in non ascii characters
    other than polish diacritical letters - it means they're being excluded
    from searching.

    :param texts: list
        List of texts
    :return: list
        Texts with non ascii characters

    :author Marcin Rybiński
    """

    polish_diacritical_letters = ['ą', 'ć', 'ę', 'ł', 'ń', 'ó', 'ś', 'ź', 'ż']
    polish_diacritical_letters += [letter.upper() for letter in polish_diacritical_letters]

    search_diacritical_letters_regex = re.compile('|'.join(polish_diacritical_letters))
    search_non_ascii_regex = re.compile(r'[ -~]')

    # first we clean text from polish polish_diacritical_letters
    texts_non_ascii = [search_non_ascii_regex.sub('', text) for text in texts]

    # and then we search for non ascii characters
    texts_non_ascii = [search_diacritical_letters_regex.sub('', text) for text in texts_non_ascii]

    return np.array(texts_non_ascii)


def search_text_with_word(texts, words, logic='or', return_index=False):
    """
    Finds the texts that contains the words

    :param texts: list
        Texts to be searched in list
    :param words: liust
        Words to searched
    :param logic: str, optional, default = 'or'
        Logic for search:
            'or' - returns texts which contains atleast one of words in param: words
            'and' - returns text which contains all words in param: words
    :param return_index: bool, optional, default = False
        Returns indices of found texts
    :return: list or tuple(list, list)

    :author Marcin Rybiński
    """
    words = [rf'\b{word}\b' for word in words]
    if logic == 'or':
        regex_pattern = r'|'.join(words)
    elif logic == 'and':
        regex_pattern = r'/^'
        regex_pattern += ''.join([rf'(?=.*{word})' for word in words])
        regex_pattern += r'.*$/m'
    regex = re.compile(regex_pattern)
    texts_found = np.array([(text, idx) for idx, text in enumerate(texts) if regex.search(text)])
    if return_index:
        return texts_found
    else:
        return texts_found[:, 0]


class NonAsciiCharacterWhitespaceSurrounder(BaseEstimator, TransformerMixin):
    """
    Surrounds non-ascii character with spaces for easier tokenization.

    Parameters
    ----------
    search_non_ascii_pattern: str, default = r'[^\x00-\x7F]'
        Pattern to search for non ascii character

    Attributes
    ----------
    search_non_ascii_pattern: str
        Pattern to search for non ascii character

    search_non_ascii_regex: compiled regex
        Compiled pattern to search for non ascii character

    Author
    ----------
    Marcin Rybiński
    """

    def __init__(self, search_non_ascii_pattern=r'[^\x00-\x7F]'):
        """ Initialize self """
        self.search_non_ascii_pattern = search_non_ascii_pattern
        self.search_non_ascii_regex = re.compile(search_non_ascii_pattern)

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Transforms given texts, sourrounding non ascii characters with whitespaces

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed text
        """
        return [self._add_whitespaces(text) for text in X]

    def _add_whitespaces(self, text):
        """
        Add white spaces around non ascii characters in text

        Parameters
        ----------
        text: str
            Single text we want to transform

        Returns
        -------
        str
            Text with added whitespaces around non ascii characters
        """

        text_list = list(text)
        non_ascii_found = self.search_non_ascii_regex.finditer(text)
        for match in non_ascii_found:
            idx = match.start()
            non_ascii_character = match.group()
            text_list[idx] = f' {non_ascii_character} '
        return ''.join(text_list)


class WordCaser(BaseEstimator, TransformerMixin):
    """
    Changes the case of letters in text

    Parameters
    ----------
    case: str, {'lower', 'upper'}, default = 'lower'
        Texts' size

    Attributes
    ----------
    case: str
        Texts' size

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self, case):
        """ Initialize self """
        possible_cases = ['upper', 'lower']
        if case not in possible_cases:
            error_message = f'Mozliwe tylko {" ".join(possible_cases)}'
            raise NotImplementedError(error_message)

        self.case = case

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Changing the case of text

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        if self.case == 'upper':
            X_transformed = [text.upper() for text in X]
        elif self.case == 'lower':
            X_transformed = [text.lower() for text in X]
        return X_transformed


class PolishLetterReplacer(BaseEstimator, TransformerMixin):
    """
    Changes polish diacritical letters to normal letters

    Attributes
    ----------
    polish_letters_replacement: dict
        Polish diacritical letters and their normal counterparts

    Author
    ----------
    Marcin Rybiński
    """

    def __init__(self):
        """ Initialize self """
        self.polish_letters_replacement = {
            'ę': 'e', 'Ę': 'E',
            'ó': 'o', 'Ó': 'O',
            'ą': 'a', 'Ą': 'A',
            'ś': 's', 'Ś': 'S',
            'ł': 'l', 'Ł': 'L',
            'ż': 'z', 'Ż': 'Z',
            'ź': 'z', 'Ź': 'Z',
            'ć': 'c', 'Ć': 'C',
            'ń': 'n', 'Ń': 'N'
        }

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Changing polish diacritical letters to normal letters

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        return [self._replace_polish_letters(text) for text in X]

    def _replace_polish_letters(self, text):
        """
        Change polish diacritical letters in text

        Parameters
        ----------
        text: str
            Single text we want to transform

        Returns
        -------
        str
            Text with changed letters to normal
        """
        for pol, rep in self.polish_letters_replacement.items():
            text = re.sub(pol, rep, text)
        return text


class NonAsciiCharactersRemover(BaseEstimator, TransformerMixin):
    """
    Deletes non ascii characters in texts

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self):
        """ Initialize self """
        pass

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Deletes non ascii characters in texts

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        return [text.encode('ascii', 'ignore').decode('ascii') for text in X]


class PunctuationRemover(BaseEstimator, TransformerMixin):
    """
    Removes punctuaction from texts

    Uses spacy PUNCT, LIST_QUOTES, LIST_ELLIPSES and LIST_HYPHENS to
    define default list of punctuation to remove.

    Parameters
    ----------
    additional_punctuation: list
        List of additional punctuation

    Attributes
    ----------
    additional_punctuation: list
        List of additional punctuation

    punctuation: list
        List of all punctuation

    punctuation_regex: regex
        Compiled regex to search for punctuation

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self, additional_punctuation=None):
        """ Initialize self """
        full_punctuation = PUNCT.split('|')
        full_punctuation += LIST_QUOTES
        full_punctuation += LIST_ELLIPSES
        full_punctuation += LIST_HYPHENS

        if additional_punctuation:
            if isinstance(additional_punctuation, list):
                full_punctuation += additional_punctuation
            else:
                raise ValueError('additional_punctuation musi byc listą')

        regex = '|'.join(full_punctuation)
        regex = re.compile(regex)

        self.additional_punctuation = additional_punctuation
        self.punctuation = full_punctuation
        self.punctuation_regex = regex

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Deletes punctuation in texts

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        return [re.sub(self.punctuation_regex, ' ', text) for text in X]


class StopWordsRemover(BaseEstimator, TransformerMixin):
    """
    Removes stop words from texts

    Uses spacy STOP_WORDS to define default list of stop words to remove.

    Parameters
    ----------
    additional_stop_words: list
        List of additional stop words

    Attributes
    ----------
    additional_stop_words: list
        List of additional stop words

    stop_words: list
        List of all stop words

    stop_words_regex: regex
        Compiled regex to search for stop words

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self, additional_stop_words=None):
        """ Initialize self """
        full_stop_words = list(STOP_WORDS)
        full_stop_words += PolishLetterReplacer().fit_transform(STOP_WORDS)

        if additional_stop_words:
            if isinstance(additional_stop_words, list):
                full_stop_words += additional_stop_words
            else:
                raise ValueError('additional_stop_words musi byc listą')

        regex = '|'.join([rf'\b{sw}\b' for sw in full_stop_words])
        regex = re.compile(regex)

        self.additional_stop_words = additional_stop_words
        self.stop_words = full_stop_words
        self.stop_words_regex = regex

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Deletes stop words in texts

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        return [re.sub(self.stop_words_regex, '', text) for text in X]


class CurrencyRemover(BaseEstimator, TransformerMixin):
    """
    Removes currency from texts

    Uses spacy CURRENCY to define default list of currency to remove.

    Parameters
    ----------
    additional_currency: list
        List of additional currency

    Attributes
    ----------
    additional_currency: list
        List of additional currencies

    currency: list
        List of all currencies

    number_with_text_detect_regex: regex
        Compiled regex to search for text that contains number

    number_with_text_split_regex: regex
        Compiled regex to split text from numbers

    currency_regex: regex
        Compiled regex to search for currency

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self, additional_currency=None):
        """ Initialize self """
        regex = CURRENCY
        regex += r'|\bzl\b|\bzł\b'
        full_currency = regex.split('|')

        if additional_currency:
            if isinstance(additional_currency, list):
                full_currency += additional_currency
                regex += '|'.join([rf'\b{c}\b' for c in additional_currency])
            else:
                raise ValueError('additional_currency musi byc listą')

        self.additional_currency = additional_currency
        self.currency = full_currency
        self.number_with_text_detect_regex = re.compile(r'[0-9]+ ?[a-zA-Z]+')
        self.number_with_text_split_regex = re.compile(r'(?<=[0-9]) ?(?=[a-zA-Z])')
        self.currency_regex = re.compile(regex)

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Deletes currency in texts

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        X_transformed = deepcopy(X)
        for idx, text in enumerate(X):
            if re.search(self.number_with_text_detect_regex, text):
                text_split = re.split(self.number_with_text_split_regex, text)
                text_join = ' '.join(text_split)
                X_transformed[idx] = text_join

        X_transformed = [re.sub(self.currency_regex, '', text) for text in X_transformed]
        return X_transformed


class UnitsRemover(BaseEstimator, TransformerMixin):
    """
    Removes units from texts

    Uses spacy UNITS to define default list of units to remove.

    Parameters
    ----------
    additional_units: list
        List of additional units

    Attributes
    ----------
    additional_units: list
        List of additional units

    units: list
        List of all units

    number_with_text_detect_regex: regex
        Compiled regex to search for text that contains number

    number_with_text_split_regex: regex
        Compiled regex to split text from numbers

    units_regex: regex
        Compiled regex to search for units

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self, additional_units=None):
        """ Initialize self """
        full_units = UNITS.split('|')
        regex = '|'.join([rf'\b{c}\b' for c in full_units])

        if additional_units:
            if isinstance(additional_units, list):
                full_units += additional_units
                regex += '|'.join([rf'\b{c}\b' for c in additional_units])
            else:
                raise ValueError('additional_currency musi byc listą')

        self.additional_units = additional_units
        self.units = full_units
        self.number_with_text_detect_regex = re.compile(r'[0-9]+ ?[a-zA-Z]+')
        self.number_with_text_split_regex = re.compile(r'(?<=[0-9]) ?(?=[a-zA-Z])')
        self.units_regex = re.compile(regex)

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Deletes units in texts

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        X_transformed = deepcopy(X)
        for idx, text in enumerate(X):
            if re.search(self.number_with_text_detect_regex, text):
                text_split = re.split(self.number_with_text_split_regex, text)
                text_join = ' '.join(text_split)
                X_transformed[idx] = text_join

        X_transformed = [re.sub(self.units_regex, '', text) for text in X_transformed]
        return X_transformed


class NumberReplacer(BaseEstimator, TransformerMixin):
    """
    Removes numbers from text

    Parameters
    ----------
    action: str, {'replacement', 'delete'}, default = 'replacement'
        How to deal with numbers:
        'replacement' - replace them with param: replacement_str
        'delete' - deletes numbers from texts

    replacement_str: str, default = 'NUMBER'
        Replacement texts for numbers. Only works with param: action='replacement'

    Attributes
    ----------
    action: list
        How to deal with numbers

    replacement_str: list
        Replacement texts for numbers

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self, action='replacement', replacement_str='NUMBER'):
        """ Initialize self """
        self.action = action
        self.replacement_str = replacement_str

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Performs action on number in text

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        X_transformed = []
        for text in X:
            if self.action == 'replacement':
                text_transformed = [f'#{self.replacement_str}' if word.isnumeric() else word
                                    for word in text.split()]
            elif self.action == 'delete':
                text_transformed = [word for word in text.split() if not word.isnumeric()]
            else:
                raise NotImplementedError()
            X_transformed.append(' '.join(text_transformed))
        return X_transformed


class WhitespaceRemover(BaseEstimator, TransformerMixin):
    """
    Removes double or more spaces from and strips whitespaces around text

    Attributes
    ----------
    whitespace_regex: compiled regex,
        Regex to search for double or more spaces in text

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self):
        """ Initialize self """
        regex = re.compile(r'\s\s+')
        self.whitespace_regex = regex

    def fit(self, X, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Removes double or more spaces from and strips whitespaces around text

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """
        X_transformed = [text.strip() for text in X]
        X_transformed = [re.sub(self.whitespace_regex, ' ', text) for text in X_transformed]
        return X_transformed


class PolBERTFeatureConstructor(BaseEstimator, TransformerMixin):
    """
    Creates features from last hidden state in PolBERT

    Attributes
    ----------
    bert_name: str
        PolBERT's name

    model: BertModel
        PolBERT model

    tokenizer: BertTokenizer
        PolBERT tokenizer

    Author
    ----------
    Marcin Rybiński
    """
    def __init__(self):
        self.bert_name = 'dkleczek/bert-base-polish-uncased-v1'
        self.model = tfr.BertModel.from_pretrained(self.bert_name)
        self.tokenizer = tfr.BertTokenizer.from_pretrained(self.bert_name)

    def fit(self, X=None, y=None):
        """
        This class don't have to be fitted, but the method fit and arguments X and y
        remains for compatibility with sklearn pipeline methods.

        Returns
        -------
        self
           Fitted estimator.
        """
        return self

    def transform(self, X, y=None):
        """
        Creates features from last hidden state in PolBERT

        Parameters
        ----------
        X : list, numpy array
            Texts we want to transform.

        Returns
        -------
        list
           Transformed texts
        """

        # we add the special tokens needed for sentence classifications
        # (these are [CLS] at the first position, and [SEP] at the end of the sentence).
        X_tokenized = [self.tokenizer.encode(text, add_special_tokens=True)
                       for text in X]

        # make all the vectors the same size
        # by padding shorter sentences with the token id 0
        max_len = max(list(map(len, X_tokenized)))
        X_padded = np.array([token + [0] * (max_len - len(token))
                             for token in X_tokenized])

        # We need to create another variable to tell it to
        # ignore (mask) the padding we've added
        attention_mask = np.where(X_padded != 0, 1, 0)

        input_ids = torch.tensor(X_padded)
        attention_mask = torch.tensor(attention_mask)

        # pass through PolBERT
        with torch.no_grad():
            last_hidden_states = self.model(input_ids, attention_mask=attention_mask)

        # we ignore all except the first vector
        # (the one associated with the [CLS] token)
        # because we want to classify the sentence
        features = last_hidden_states[0][:, 0, :].numpy()

        return features

from sklearn.preprocessing import LabelEncoder
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import wordpunct_tokenize
import torchtext as text
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.functional import one_hot
import numpy as np
from torch.nn.utils.rnn import pad_sequence

lemmatizer = WordNetLemmatizer()


def split(dataframe):
    """Split dataframe into X, y

    :param dataframe: pd.Dataframe containing label/input columns
    :return: dataframe['input'], dataframe['label']
    """
    assert (
        "input" in dataframe.columns and "label" in dataframe.columns
    ), "There are no input/label columns in dataframe. Found only {dataframe.columns}"

    return dataframe["input"], dataframe["label"]


def encode_txt2int(data):
    """Encode different text to different integers 0, 1, ...

    :param data: dataframe, pd.series or numpy.ndarray
    :return: encoded values as numpy array
    """

    enc = LabelEncoder().fit(data)
    data_num = enc.transform(data)

    return data_num


def lemmatize_series(df):
    """Lemmatize words in pd.series

    :param df: pd.series
    :return: lemmatized pd.series
    """

    df_tokenized = df.apply(lambda setence: wordpunct_tokenize(setence))

    def lemmatize_list(words):
        """Lemmatize words in pd.series

        :param words: list/iterable object with string as an item
        :return: list of lemmatized words
        """

        lemmatized_words = []

        for word in words:
            lemmatized_words.append(lemmatizer.lemmatize(word).lower())

        return lemmatized_words

    df_lemmatized = df_tokenized.apply(lambda words: lemmatize_list(words))

    return df_lemmatized


def vectorize_dataseries(df):
    """Embed words in 300 dimensional space with GloVe

    :param df: pd.series with each element as a list (or iterable object) of words
    :return: pd.series with vectors as embedded words
    """

    vec = text.vocab.GloVe(name="6B", dim=300)

    def vectorize(words):
        """Embed words in pd.series

        :param words: list/iterable object with string as an item
        :return: list of embedded words
        """

        return vec.get_vecs_by_tokens(words)

    df_num = df.apply(lambda words: vectorize(words))

    return df_num


class Emotions(Dataset):
    """Class implementing and inheriting from torch.utils.data.Dataset
    Emotions implements mainly emotions dataset consiting of string as an input

    :param inputs: list/torch.tensor/numpy.array with a sequence of vectors
    :param labels: list/torch.tensor/numpy.array of integers - target for the corresponding input in the form of Integer
    :param num_classes: number of different classes in total
    """

    def __init__(self, inputs, labels, num_classes):
        self.inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
        self.labels = labels
        self.num_classes = num_classes

    def __len__(self):
        assert len(self.inputs) == len(
            self.labels
        ), f"Length of inputs ({len(self.inputs)}) and labels ({len(self.labels)}) lists don't match"
        return len(self.inputs)

    def __getitem__(self, index):
        input = self.inputs[index]
        label = torch.tensor(self.labels[index])

        return input, one_hot(label, self.num_classes)

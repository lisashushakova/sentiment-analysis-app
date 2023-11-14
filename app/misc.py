import enum
import re

import nltk
import numpy as np
from gensim.models import Word2Vec
from keras_preprocessing.sequence import pad_sequences
from navec import Navec


from keras.preprocessing.text import Tokenizer

class DatasetType(enum.Enum):
    BANKS = 'Banks'
    MEDIA = 'Mass Media'
    TWITTER = 'Twitter'


class ClassifierType(enum.Enum):
    LOGISTIC_REGRESSION = 'Logistic Regression'
    NAIVE_BAYESIAN = 'Naive Bayesian'
    CNN = 'CNN'
    RNN = 'RNN'


class WordVectorizationType(enum.Enum):
    OHE = 'One-hot-encoding'
    W2V = 'Word2vec'


class TextVectorizationType(enum.Enum):
    BOW = 'Bag-of-words'
    TF_IDF = 'TF-IDF'


class Word2VecModelType(enum.Enum):
    NAVEC = 'Navec'
    TRAINED = 'Trained'


class mean_vectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.dim = dim

    def fit(self, X):
        return self

    def transform(self, X):
        res = []
        for i, words in enumerate(X):
            res.append(np.mean([self.word2vec[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)], axis=0))
        return np.array(res)



from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict


class tfidf_vectorizer(object):
    def __init__(self, word2vec, dim):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = dim

    def fit(self, X):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        res = []

        for i, words in enumerate(X):

            res.append(np.mean(
                [self.word2vec[w] * self.word2weight[w] for w in words if w in self.word2vec] or [np.zeros(self.dim)],
                axis=0))

        return np.array(res)


class nn_vectorizer(object):

    def __init__(self, max_features, length):
        self.tokenizer = Tokenizer(num_words=max_features)
        self.length = length

    def fit(self, X):
        self.tokenizer.fit_on_texts(X)
        return self

    def transform(self, X):
        return pad_sequences(self.tokenizer.texts_to_sequences(X), self.length)




from keras import backend as K

def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision.

    Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall.

    Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    return 2 * ((precision_val * recall_val) / (precision_val + recall_val + K.epsilon()))


def preprocess_text(text):
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    from pymorphy3 import MorphAnalyzer

    nltk.download('punkt')
    nltk.download('stopwords')

    punctuation_marks = ['!', ',', '(', ')', ':', '-', '?', '.', '..', '...', '«', '»', ';', '–', '--']
    stop_words = stopwords.words("russian")
    morph = MorphAnalyzer()

    res = text.lower()

    res = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', res)
    res = re.sub('@[^\s]+', 'USER', res)

    res = re.sub(' +', ' ', res).strip()
    res = res.replace('ё', 'е')
    res = re.sub('[^a-zA-Zа-яА-Я1-9]+', ' ', res)

    tokens = word_tokenize(res)
    preprocessed_text = [t for t in tokens if t not in punctuation_marks and t not in stop_words]
    preprocessed_text = [morph.parse(t)[0].normal_form for t in preprocessed_text]

    return preprocessed_text


def select_vocabulary(w2v_model_type, dataset):
    if w2v_model_type == Word2VecModelType.NAVEC:
        DIM = 300
        return Navec.load('../navec_hudlit_v1_12B_500K_300d_100q.tar'), DIM
    elif dataset == DatasetType.BANKS:
        DIM = 200
        return Word2Vec.load('../embeddings/banks_model/banks.model').wv, DIM
    elif dataset == DatasetType.MEDIA:
        DIM = 200
        return Word2Vec.load('../embeddings/mass_media_model/mass_media.model').wv, DIM
    elif dataset == DatasetType.TWITTER:
        DIM = 200
        return Word2Vec.load('../embeddings/tweets_model/tweets_model.w2v').wv, DIM

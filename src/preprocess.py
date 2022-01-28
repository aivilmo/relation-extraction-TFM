#!/usr/bin/env python

from cmath import nan
from typing_extensions import Self
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class Preprocessor:
    _instance = None

    def __new__(cls: type[Self], *args, **kwargs) -> Self:
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self._le = LabelEncoder()
        self._ohe = OneHotEncoder(sparse=False)

    def train_test_split(self, train_df: pd.DataFrame, test_df: pd.DataFrame):
        train_df_old, test_df_old = train_df.copy(), test_df.copy()
        train_df = train_df.drop("tag", axis=1).drop("filename", axis=1)
        test_df = test_df.drop("tag", axis=1).drop("filename", axis=1)

        X_train = self._ohe.fit_transform(train_df.word.values.reshape(-1, 1))
        y_train = self._le.fit_transform(train_df_old.tag.values).reshape(-1, 1)

        X_test = self._ohe.transform(test_df.word.values.reshape(-1, 1))
        y_test = self._le.transform(test_df_old.tag.values).reshape(-1, 1)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def preprocess(text: str) -> list:
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk.corpus import stopwords
        import re
        import unidecode

        unaccented_string = unidecode.unidecode(text)
        alphanumeric_text = re.sub("[^0-9a-zA-Z]+", " ", unaccented_string)

        # Conversión texto a minúsculas y tokenización a lista de palabras
        tokens = alphanumeric_text.lower().split()

        # Eliminación las stopwords
        stop_words = set(stopwords.words("spanish"))
        tokens_without_stopwords = [
            token for token in tokens if token not in stop_words
        ]

        # Lematización
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens_lemmas = [
            wordnet_lemmatizer.lemmatize(token) for token in tokens_without_stopwords
        ]

        # Stemming
        stemmer = PorterStemmer()
        tokens_stemmed = [stemmer.stem(token) for token in tokens_lemmas]

        return " ".join(tokens_stemmed)

    @staticmethod
    def sentences_tokenizer(text: str) -> list:
        from nltk.tokenize import sent_tokenize

        return sent_tokenize(text)

    @staticmethod
    def process_relations(relations: dict) -> pd.DataFrame:
        import numpy as np

        df: pd.DataFrame = pd.DataFrame()
        for entities, relation in relations.items():
            relation = pd.DataFrame(
                [
                    {
                        "word1": Preprocessor.preprocess(entities[0]),
                        "tag1": entities[1],
                        "word2": Preprocessor.preprocess(entities[2]),
                        "tag2": entities[3],
                        "relation": relation,
                        "sentence": np.nan,
                    }
                ]
            )
            df = df.append(relation)
        return df

    @staticmethod
    def process_content(df: pd.DataFrame, content: str):
        from nltk.util import ngrams

        for sentence in Preprocessor.sentences_tokenizer(content):
            processed_sentences: list = Preprocessor.preprocess(sentence).split()

            sentence_ngrams: list = (
                processed_sentences
                + [" ".join(bigram) for bigram in ngrams(processed_sentences, 2)]
                + [" ".join(trigram) for trigram in ngrams(processed_sentences, 3)]
            )
            df.loc[
                df.eval(
                    "word1 in @sentence_ngrams and word2 in @sentence_ngrams and sentence.isnull()"
                ),
                "sentence",
            ] = " ".join(processed_sentences)

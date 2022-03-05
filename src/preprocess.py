#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import time


class Preprocessor:

    _n_classes = 8
    _instance = None

    @staticmethod
    def instance():
        if Preprocessor._instance == None:
            Preprocessor()
        return Preprocessor._instance

    def __init__(self) -> None:
        if Preprocessor._instance != None:
            raise Exception

        Preprocessor._instance = self

    @staticmethod
    def train_test_split(
        train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:
        from featureshandler import FeaturesHandler

        # Transform labels
        y_train, y_test = Preprocessor.encode_labels(train_df, test_df)

        train_df.drop(y_column, axis=1, inplace=True)
        test_df.drop(y_column, axis=1, inplace=True)

        # Handle features from data
        X_train = FeaturesHandler.instance().handleFeatures(train_df)
        X_test = FeaturesHandler.instance().handleFeatures(test_df, test=True)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def preprocess(text: str, stopwords: bool = False) -> str:
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk.corpus import stopwords
        import re
        import unidecode

        unaccented_string = unidecode.unidecode(text)
        alphanumeric_text = re.sub("[^0-9a-zA-Z]+", " ", unaccented_string)

        # Conversión texto a minúsculas y tokenización a lista de palabras
        tokens = alphanumeric_text.lower().split()

        # Eliminación las stopwords
        if stopwords:
            stop_words = set(stopwords.words("spanish"))
            tokens = [token for token in tokens if token not in stop_words]

        # Lematización
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens_lemmas = [wordnet_lemmatizer.lemmatize(token) for token in tokens]

        # Stemming
        stemmer = PorterStemmer()
        tokens_stemmed = [stemmer.stem(token) for token in tokens_lemmas]

        return " ".join(tokens_stemmed)

    @staticmethod
    def process_content(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        print(f"Loaded {len(collection)} sentences for fitting.")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            for relation in sentence.relations:
                relation = pd.Series(
                    {
                        "word1": Preprocessor.preprocess(
                            relation.from_phrase.text.lower()
                        ),
                        "tag1": relation.from_phrase.label,
                        "word2": Preprocessor.preprocess(
                            relation.to_phrase.text.lower()
                        ),
                        "tag2": relation.to_phrase.label,
                        "relation": relation.label,
                        "sentence": Preprocessor.preprocess(sentence.text),
                    },
                    name=index,
                )
                index += 1
                df = df.append(relation)

        print(f"Training completed: Stored {index} relation pairs.")
        return df

    @staticmethod
    def process_content_as_IOB_format(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        print(f"Loaded {len(collection)} sentences for fitting.")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = keyphrases.text.split()
                for i in range(len(entities)):
                    tag = "B-" if i == 0 else "I-"
                    sentence_entities[entities[i]] = tag + keyphrases.label

            words = sentence.text.split()
            for word in words:
                tag = sentence_entities.get(word, "O")
                word = pd.Series(
                    {"word": word, "tag": tag},
                    name=index,
                )
                index += 1
                df = df.append(word)

        print(f"Training completed: Stored {index} words.")
        return df

    @staticmethod
    def prepare_labels(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        from keras.utils.np_utils import to_categorical

        return to_categorical(
            y_train, num_classes=Preprocessor._n_classes
        ), to_categorical(y_test, num_classes=Preprocessor._n_classes)

    @staticmethod
    def encode_labels(
        train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:
        from sklearn.preprocessing import LabelEncoder
        from coremodel import CoreModel

        # Remove classes what are in test but not in train
        Preprocessor.remove_invalid_classes(test_df, train_df, y_column)

        # Remove classes not in test so we cant test it (e.g. I-Reference)
        Preprocessor.remove_invalid_classes(train_df, test_df, y_column)

        print(f"Transforming {y_column} into labels")
        le = LabelEncoder()
        y_train = le.fit_transform(train_df[y_column].values)
        y_test = le.transform(test_df[y_column].values)
        CoreModel.instance().set_labels(list(le.classes_))

        return y_train, y_test

    @staticmethod
    def remove_invalid_classes(
        df_to_remove: pd.DataFrame, df_from_remove: pd.DataFrame, y_column: str
    ) -> None:
        invalid_labels: list = list(
            set(df_to_remove[y_column].unique())
            - set(df_from_remove[y_column].unique())
        )

        df_to_remove.drop(
            df_to_remove.loc[df_to_remove[y_column].isin(invalid_labels)].index,
            inplace=True,
        )

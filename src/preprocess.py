#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np
import time
from logger import Logger

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)


class Preprocessor:

    _n_classes = None
    _instance = None

    @staticmethod
    def instance():
        if Preprocessor._instance == None:
            Preprocessor()
        return Preprocessor._instance

    def __init__(self) -> None:
        if Preprocessor._instance != None:
            raise Exception

        self._logger = Logger.instance()
        Preprocessor._instance = self

    @staticmethod
    def train_test_split(
        train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:
        from featureshandler import FeaturesHandler

        # train_df = Preprocessor.filter_word_tag_ocurrences(train_df)

        Preprocessor._n_classes = len(set(train_df[y_column].unique()))

        # Transform labels
        y_train, y_test = Preprocessor.encode_labels(train_df, test_df)

        train_df.drop(y_column, axis=1, inplace=True)
        test_df.drop(y_column, axis=1, inplace=True)

        # Handle features from data
        X_train = FeaturesHandler.instance().handle_features(train_df)
        X_test = FeaturesHandler.instance().handle_features(test_df, test=True)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def data_iterator(
        df: pd.DataFrame, chunk_size: int = 1500, y_column: str = "tag"
    ) -> np.ndarray:
        from featureshandler import FeaturesHandler
        import dask.dataframe as dd
        from dask import delayed, compute

        df.drop(y_column, axis=1, inplace=True)

        delayed_results = []
        ddf = dd.from_pandas(df, chunksize=chunk_size)
        partitions = len(ddf.to_delayed())
        part_n = 1
        for part in ddf.to_delayed():
            df = part.compute()
            Preprocessor.instance()._logger.info(
                f"Computing part of dataframe, part {part_n} of {partitions}"
            )
            part_n += 1
            X = delayed(FeaturesHandler.instance().handle_features)(df)
            delayed_results.append(X)

        start = time.time()
        results = compute(*delayed_results, scheduler="threads")
        Preprocessor.instance()._logger.info(
            f"Parallel features handler: {(time.time() - start) / 60} minutes"
        )
        return results

    @staticmethod
    def preprocess(text: str, without_stopwords: bool = False) -> str:
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk.corpus import stopwords
        import re
        import unidecode

        unaccented_string = unidecode.unidecode(text)
        alphanumeric_text = re.sub("[^0-9a-zA-Z]+", " ", unaccented_string)

        # Conversión texto a minúsculas y tokenización a lista de palabras
        tokens = alphanumeric_text.lower().split()

        # Eliminación las stopwords
        if without_stopwords:
            stop_words = set(stopwords.words("spanish"))
            tokens = [token for token in tokens if token not in stop_words]
            return " ".join(tokens)

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
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )

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

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} relation pairs."
        )
        return df

    @staticmethod
    def process_content_as_IOB_format(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_IOB_format")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = keyphrases.text.split()
                for i in range(len(entities)):
                    tag = "B-" if i == 0 else "I-"
                    old_entity = sentence_entities.get(entities[i], [])
                    old_entity.append(tag + keyphrases.label)
                    sentence_entities[entities[i]] = old_entity

            words = sentence.text.split()
            for word in words:
                tag = sentence_entities.get(word, ["O"])
                word = pd.Series(
                    {"word": word, "tag": max(set(tag), key=tag.count)},
                    name=index,
                )
                index += 1
                df = df.append(word)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} words."
        )
        return df

    @staticmethod
    def process_content_as_BILUOV_format(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_BILUOV_format")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = Preprocessor.preprocess(keyphrases.text).split()
                for i in range(len(entities)):
                    # 1 word entity
                    if len(entities) == 1:
                        tag = "U-"
                    # More than 1 word entities
                    elif i == 0:
                        tag = "B-"
                    # Last word in entity
                    elif i == len(entities) - 1:
                        tag = "L-"
                    # Inner word
                    else:
                        tag = "I-"
                    # Overlapped word
                    if sentence_entities.get(entities[i], -1) != -1:
                        tag = "V-"
                    sentence_entities[entities[i]] = tag + keyphrases.label

            words = Preprocessor.preprocess(sentence.text).split()
            for word in words:
                tag = sentence_entities.get(word, "O")
                word = pd.Series(
                    {"word": word, "tag": tag},
                    name=index,
                )
                index += 1
                df = df.append(word)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} words."
        )
        return df

    @staticmethod
    def process_content_as_sentences(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection
        from embeddinghandler import Embedding, TransformerEmbedding

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_sentences")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer()

        for sentence in collection.sentences:
            sent = TransformerEmbedding.instance().sentence_vector(sentence.text)
            tokenized_sent = TransformerEmbedding.instance().tokenize(sentence.text)
            # tokenized_sent = ["[CLS]"] + tokenized_sent + ["[SEP]"]
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = keyphrases.text.split()
                for i in range(len(entities)):
                    entity_word = TransformerEmbedding.instance().tokenize(entities[i])
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + keyphrases.label
                    for j in range(len(entity_word)):
                        if j != 0:
                            break
                        sentence_entities[entity_word[j]] = tag
            for i in range(len(tokenized_sent)):
                tag = sentence_entities.get(tokenized_sent[i], "O")
                word = pd.Series(
                    {"token": tokenized_sent[i], "vector": sent[i + 1], "tag": tag},
                    name=index,
                )
                index += 1
                df = df.append(word)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} words."
        )
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
        # Preprocessor.remove_invalid_classes(test_df, train_df, y_column)

        # Remove classes not in test so we cant test it (e.g. I-Reference)
        # Preprocessor.remove_invalid_classes(train_df, test_df, y_column)

        Preprocessor.instance()._logger.info(f"Transforming {y_column} into labels")
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

    @staticmethod
    def filter_word_tag_ocurrences(df: pd.DataFrame) -> pd.DataFrame:
        from random import sample

        print(df.value_counts())
        print(df.tag.value_counts())

        # Other posibilities
        # train_df = train_df.groupby("word").filter(lambda x: len(x) < 700)
        # train_df.drop_duplicates(
        #     subset=["word", "tag"], keep="last", inplace=True, ignore_index=True
        # )
        # o_indexes = df[(df.tag == "O") & (df.word == "de")].index.tolist()

        # o_indexes = df[df.tag == "O"].index.tolist()
        # indexes_to_drop = sample(o_indexes, int(len(o_indexes) * 0.6))
        # indexes_to_keep = set(range(df.shape[0])) - set(indexes_to_drop)
        # df = df.take(list(indexes_to_keep))

        # print("AFTER")
        # print(df.value_counts())
        # print(df.tag.value_counts())
        return df

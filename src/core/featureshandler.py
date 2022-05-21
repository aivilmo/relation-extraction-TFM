#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from core.embeddinghandler import Embedding, WordEmbedding, TransformerEmbedding
from logger.logger import Logger


class FeaturesHandler:

    _instance = None
    _logger = Logger.instance()

    _transformers: list = [
        "distilbert-base-uncased",
        "distilbert-base-cased",
        "bert-base-uncased",
        "bert-base-cased",
        "bert-base-multilingual-uncased",
        "bert-base-multilingual-cased",
        "dccuchile/bert-base-spanish-wwm-uncased",
        "dccuchile/bert-base-spanish-wwm-cased",
        "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "ixa-ehu/ixambert-base-cased",
        "gpt2",
    ]
    _features_for_task: dict = {
        "scenario2-taskA": [
            "sent_emb",
            "bag_of_words",
            "single_word_emb",
            "tf_idf",
            "chars",
            "tokens",
            "seq2seq",
        ]
        + _transformers,
        "scenario3-taskB": ["with_entities", "word_emb"] + _transformers,
    }

    @staticmethod
    def instance():
        if FeaturesHandler._instance is None:
            FeaturesHandler()
        return FeaturesHandler._instance

    def __init__(self, task: str) -> None:
        if FeaturesHandler._instance is not None:
            return

        self._task: str = task

        self._le: LabelEncoder = LabelEncoder()
        self._cv: CountVectorizer = CountVectorizer()
        self._tf: TfidfVectorizer = TfidfVectorizer()
        self._ck: Tokenizer = Tokenizer(char_level=True)
        self._tk: Tokenizer = Tokenizer()

        self._features: list = ["PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"]
        self._is_transformer: bool = False

        FeaturesHandler._instance = self

    @property
    def features(self) -> list:
        return self._features

    @features.setter
    def features(self, features: str) -> None:
        self._features = features
        self._is_transformer = (
            list(
                filter(
                    lambda x: x in features,
                    self._transformers,
                )
            )
            != []
        )

    def check_features_for_task(self) -> None:
        for feat in self._features:
            if feat not in self._features_for_task[self._task]:
                import sys

                self._logger.info(f"Feature {feat} don't match with task {self._task}")
                self._logger.info(
                    f"Available features for task {self._task}: {self._features_for_task[self._task]}"
                )
                sys.exit()

    def handle_features(self, df: pd.DataFrame, test: bool = False) -> np.ndarray:
        columns = []
        self._logger.info(
            "Handling features: " + ", ".join(self.features) + f" for task {self._task}"
        )
        self.check_features_for_task()

        # NER features
        if "tf_idf" in self._features:
            self._feat_tf_idf(df, test=test)
            columns += ["token"]
        if "bag_of_words" in self._features:
            self._feat_bag_of_words(df, test=test)
            columns += ["token"]
        if "single_word_emb" in self._features:
            self._feat_single_word_emb(df)
            columns += ["token"]
        if "sent_emb" in self._features:
            self._feat_sent_emb(df)
            columns += ["sentence"]
        if "chars" in self._features:
            self._feat_chars(df, test=test)
            columns += ["token"]
        if "tokens" in self._features:
            self._feat_tokens(df, test=test)
            columns += ["token"]
        if "seq2seq" in self._features:
            self._feat_seq2seq(df)
            return

        # RE fearures
        if "with_entities" in self._features:
            self._feat_with_tags(df, test)
            columns += ["tag1"] + ["tag2"]
        if "word_emb" in self._features:
            self._feat_word_emb(df)
            columns += ["token1"] + ["token2"]

        # Both tasks
        if self._is_transformer:
            self._feat_transformer(df)
            columns += ["vector"] if "taskA" in self._task else ["token1"] + ["token2"]

        features: np.ndarray = FeaturesHandler._combine_features(df, columns)
        self._logger.info(
            f"Features matrix successfully generated, with data shape: {features.shape}"
        )
        return np.nan_to_num(features)

    def _fit_on_feature_extraction(
        self, df: pd.DataFrame, object, column: str, test: bool = False
    ) -> pd.DataFrame:
        if not test:
            self._logger.info(f"Fitting words to {object}")
            object.fit(df[column].unique().tolist())
            self._logger.info(f"Vocab size: {len(object.vocabulary_.keys())}")

        df[column] = df[column].apply(
            lambda x: object.transform([x]).toarray().reshape(-1)
        )
        return df

    def _feat_tf_idf(
        self, df: pd.DataFrame, column: str = "token", test: bool = False
    ) -> None:
        df = self._fit_on_feature_extraction(df, self._tf, column, test)

    def _feat_bag_of_words(
        self, df: pd.DataFrame, column: str = "token", test: bool = False
    ) -> None:
        df = self._fit_on_feature_extraction(df, self._cv, column, test)

    @staticmethod
    def _feat_single_word_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            WordEmbedding.instance().load_model()

        df["token"] = df.token.apply(lambda x: WordEmbedding.instance().word_vector(x))

    @staticmethod
    def _feat_sent_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            tokens = Embedding.prepare_text_to_train(df)
            WordEmbedding.instance().load_model()
            WordEmbedding.instance().train_word_embedding(tokens)

        df["sentence"] = df.sentence.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )

    def _fit_on_texts(
        self, df: pd.DataFrame, object, maxlen: int, test: bool = False
    ) -> pd.DataFrame:
        if not test:
            object.fit_on_texts(df.token)
            vocab_size = len(object.word_index) + 1
            self._logger.info(f"Vocab size: {vocab_size}")

        df["token"] = df.token.apply(lambda x: object.texts_to_sequences([x])[0])
        df["token"] = df.token.apply(lambda x: pad_sequences([x], maxlen=maxlen)[0])
        self._logger.info(f"Matriz features for emebdding: {df.token.shape}")
        return df

    def _feat_chars(self, df: pd.DataFrame, test: bool = False) -> None:
        df = self._fit_on_texts(df, self._ck, 15, test)

    def _feat_tokens(self, df: pd.DataFrame, test: bool = False) -> None:
        df = self._fit_on_texts(df, self._tk, 3, test)

    def _feat_seq2seq(self, df: pd.DataFrame) -> None:
        df.rename(columns={"token": "words", "tag": "labels"}, inplace=True)

    def _feat_with_tags(self, df: pd.DataFrame, test: bool = False) -> None:
        def _fit_transform(df, tag, values, test):
            if not test:
                df[tag] = self._le.fit_transform(values)
            df[tag] = self._le.transform(values)
            return df

        df = _fit_transform(df, "tag1", df.tag1.values, test)
        df = _fit_transform(df, "tag2", df.tag2.values, test)

    @staticmethod
    def _feat_word_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            # tokens = Embedding.prepare_text_to_train(df)
            WordEmbedding.instance().load_model()
            # WordEmbedding.instance().train_word_embedding(tokens)

        df["token1"] = df.token1.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )
        df["token2"] = df.token2.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )

    def _feat_transformer(self, df: pd.DataFrame) -> None:
        if "taskA" in self._task:
            return

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(self._features[0])

        df["token1"] = df.apply(
            lambda x: TransformerEmbedding.instance().apply_transformer(x, "token1"),
            axis=1,
        )
        self._logger.info("Finished with apply_transformer for token1")

        df["token2"] = df.apply(
            lambda x: TransformerEmbedding.instance().apply_transformer(x, "token2"),
            axis=1,
        )
        self._logger.info("Finished with apply_transformer for token2")

    @staticmethod
    def _combine_features(df: pd.DataFrame, columns: list) -> np.ndarray:
        columns_to_delete = df.columns

        def stack_vectors(list_of_vect: list) -> np.ndarray:
            full_vector = np.hstack(list_of_vect)
            dim_features = full_vector.shape[0]
            full_vector.reshape(1, dim_features)
            return full_vector

        df["features"] = df[columns].values.tolist()
        df["features"] = df.features.apply(lambda x: stack_vectors(x))
        df.drop(columns_to_delete, axis=1, inplace=True)
        return np.vstack(df.values.tolist())

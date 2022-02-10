#!/usr/bin/env python

import numpy as np
import pandas as pd
from embeddinghandler import Embedding, WordEmbedding, SentenceEmbedding
from sklearn.preprocessing import LabelEncoder


class FeaturesHandler:

    _instance = None

    @staticmethod
    def instance():
        if FeaturesHandler._instance == None:
            FeaturesHandler()
        return FeaturesHandler._instance

    def __init__(self) -> None:
        if FeaturesHandler._instance != None:
            raise Exception

        self._le: LabelEncoder = LabelEncoder()
        self._features: list = ["with_entities", "word_emb", "sent_emb"]
        FeaturesHandler._instance = self

    @property
    def features(self) -> list:
        return self._features

    @features.setter
    def features(self, features: str) -> None:
        self._features = features

    def handleFeatures(self, df: pd.DataFrame, test: bool = False) -> np.ndarray:
        print("Handling features:", ", ".join(self.features))

        df.drop("relation", axis=1, inplace=True)
        columns = []

        if "with_entities" in self._features:
            FeaturesHandler._feat_with_tags(df, test)
            columns += ["tag1"] + ["tag2"]
        if "word_emb" in self._features:
            FeaturesHandler._feat_word_emb(df, test)
            columns += ["word1"] + ["word2"]
        if "sent_emb" in self._features:
            FeaturesHandler._feat_sent_emb(df, test)
            columns += ["sentence"]

        features: np.ndarray = FeaturesHandler._combine_features(df, columns)
        print("Features matrix succesfully generated, with data shape:", features.shape)
        return np.nan_to_num(features)

    def encode_labels(self):
        pass

    @staticmethod
    def _feat_with_tags(df: pd.DataFrame, test: bool = False) -> None:
        if not test:
            print("Fitting word labels")
            df["tag1"] = FeaturesHandler.instance()._le.fit_transform(df.tag1.values)
            df["tag2"] = FeaturesHandler.instance()._le.fit_transform(df.tag2.values)
            return

        df["tag1"] = FeaturesHandler.instance()._le.transform(df.tag1.values)
        df["tag2"] = FeaturesHandler.instance()._le.transform(df.tag2.values)

    @staticmethod
    def _feat_word_emb(df: pd.DataFrame, test: bool = False) -> None:
        if not test:
            tokens = Embedding.prepare_text_to_train(df)
            WordEmbedding.instance().train_word_emebdding(tokens)

        df["word1"] = df.word1.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )
        df["word2"] = df.word2.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )

    @staticmethod
    def _feat_sent_emb(df: pd.DataFrame, test: bool = False) -> None:
        if not test:
            sentences = Embedding.prepare_text_to_train(df)
            SentenceEmbedding.instance().train_sentence_emebdding(sentences)

        df["sentence"] = df.sentence.apply(
            lambda x: SentenceEmbedding.instance().sentence_to_vector(x)
        )

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

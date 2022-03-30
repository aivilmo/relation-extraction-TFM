#!/usr/bin/env python

import numpy as np
import pandas as pd
from embeddinghandler import Embedding, WordEmbedding, TransformerEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

from logger.logger import Logger


class FeaturesHandler:

    _instance = None

    @staticmethod
    def instance():
        if FeaturesHandler._instance is None:
            FeaturesHandler()
        return FeaturesHandler._instance

    def __init__(self) -> None:
        if FeaturesHandler._instance is not None:
            raise Exception

        self._le: LabelEncoder = LabelEncoder()
        self._cv: CountVectorizer = CountVectorizer()
        self._tf: TfidfVectorizer = TfidfVectorizer()
        self._ck: Tokenizer = Tokenizer(char_level=True)
        self._tk: Tokenizer = Tokenizer()

        self._features: list = ["with_entities", "word_emb", "sent_emb"]
        self._logger = Logger.instance()
        FeaturesHandler._instance = self

    @property
    def features(self) -> list:
        return self._features

    @features.setter
    def features(self, features: str) -> None:
        self._features = features

    def handle_features(self, df: pd.DataFrame, test: bool = False) -> np.ndarray:
        self._logger.info(f"Handling features: " + ", ".join(self.features))
        columns = []

        if "single_word_emb" in self._features:
            self._feat_single_word_emb(df)
            columns += ["word"]
        if "tf_idf" in self._features:
            self._feat_tf_idf(df, test=test)
            columns += ["word"]
        if "with_entities" in self._features:
            self._feat_with_tags(df, test)
            columns += ["tag1"] + ["tag2"]
        if "word_emb" in self._features:
            self._feat_word_emb(df)
            columns += ["word1"] + ["word2"]
        if "sent_emb" in self._features:
            self._feat_sent_emb(df)
            columns += ["sentence"]
        if "bag_of_words" in self._features:
            self._feat_bag_of_words(df, test=test)
            columns += ["word"]
        if "chars" in self._features:
            self._feat_chars(df, test=test)
            columns += ["word"]
        if "tokens" in self._features:
            self._feat_tokens(df, test=test)
            columns += ["word"]
        if (
            "distilbert-base-uncased" in self._features
            or "distilbert-base-cased" in self._features
            or "bert-base-uncased" in self._features
            or "bert-base-cased" in self._features
            or "bert-base-multilingual-uncased" in self._features
            or "bert-base-multilingual-cased" in self._features
            or "dccuchile/bert-base-spanish-wwm-uncased" in self._features
            or "dccuchile/bert-base-spanish-wwm-cased" in self._features
            or "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es" in self._features
            or "ixa-ehu/ixambert-base-cased" in self._features
            or "gpt2" in self._features
        ):
            columns += ["vector"]

        features: np.ndarray = FeaturesHandler._combine_features(df, columns)
        self._logger.info(
            f"Features matrix succesfully generated, with data shape: {features.shape}"
        )
        return np.nan_to_num(features)

    def _feat_with_tags(self, df: pd.DataFrame, test: bool = False) -> None:
        if not test:
            self._logger.info("Fitting word labels")
            df["tag1"] = self._le.fit_transform(df.tag1.values)
            df["tag2"] = self._le.fit_transform(df.tag2.values)
            return

        df["tag1"] = self._le.transform(df.tag1.values)
        df["tag2"] = self._le.transform(df.tag2.values)

    @staticmethod
    def _feat_word_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            tokens = Embedding.prepare_text_to_train(df)
            WordEmbedding.instance().load_model(
                "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4"
                ".bin "
            )
            WordEmbedding.instance().train_word_emebdding(tokens)

        df["word1"] = df.word1.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )
        df["word2"] = df.word2.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )

    @staticmethod
    def _feat_sent_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            tokens = Embedding.prepare_text_to_train(df)
            WordEmbedding.instance().load_model(
                "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4"
                ".bin "
            )
            WordEmbedding.instance().train_word_emebdding(tokens)

        df["sentence"] = df.sentence.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )

    def _feat_bag_of_words(
        self, df: pd.DataFrame, column: str = "word", test: bool = False
    ) -> None:
        if not test:
            self._logger.info("Fitting words to bag of words")
            self._cv.fit(df[column].unique().tolist())
            self._logger.info(f"Vocab size: {len(self._cv.vocabulary_.keys())}")

        df[column] = df[column].apply(
            lambda x: self._cv.transform([x]).toarray().reshape(-1)
        )

    @staticmethod
    def _feat_single_word_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            WordEmbedding.instance().load_model(
                "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4"
                ".bin "
            )

        df["word"] = df.word.apply(lambda x: WordEmbedding.instance().word_vector(x))

    def _feat_tf_idf(
        self, df: pd.DataFrame, column: str = "word", test: bool = False
    ) -> None:
        if not test:
            self._logger.info("Fitting words to tf idf")
            self._tf.fit(df[column].unique().tolist())
            self._logger.info(f"Vocab size: {len(self._tf.vocabulary_.keys())}")

        df[column] = df[column].apply(
            lambda x: self._tf.transform([x]).toarray().reshape(-1)
        )

    def _feat_chars(self, df: pd.DataFrame, test: bool = False) -> None:
        if not test:
            self._ck.fit_on_texts(df.word)
            vocab_size = len(self._ck.word_index) + 1
            self._logger.info(f"Vocab size: {vocab_size}")

        df["word"] = df.word.apply(lambda x: self._ck.texts_to_sequences([x])[0])
        df["word"] = df.word.apply(lambda x: pad_sequences([x], maxlen=15)[0])
        self._logger.info(f"Matriz features for emebdding: {df.word.shape}")

    def _feat_tokens(self, df: pd.DataFrame, test: bool = False) -> None:
        if not test:
            self._tk.fit_on_texts(df.word)
            vocab_size = len(self._tk.word_index) + 1
            self._logger.info(f"Vocab size: {vocab_size}")

        df["word"] = df.word.apply(lambda x: self._tk.texts_to_sequences([x])[0])
        df["word"] = df.word.apply(lambda x: pad_sequences([x], maxlen=3)[0])
        self._logger.info(f"Matriz features for emebdding: {df.word.shape}")

    def _feat_transformer(self, df: pd.DataFrame, type) -> None:
        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=type)

        df["word"] = df.word.apply(
            lambda x: TransformerEmbedding.instance().word_vector(x)
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

    @staticmethod
    def _combine_2D_features(df: pd.DataFrame, columns: list) -> np.ndarray:
        columns_to_delete = df.columns

        def stack_vectors(list_of_vect: list) -> np.ndarray:
            full_vector = np.hstack(list_of_vect)
            dim_features = full_vector.shape[0]
            # For 2D features vectors
            dim_of_features = full_vector.shape[1]
            full_vector.reshape(1, dim_features, dim_of_features)
            return full_vector

        def pad_vector(vector: np.ndarray, n_rows: int = 100) -> np.ndarray:
            for _ in range(n_rows - vector.shape[0]):
                vector = np.concatenate((vector, np.zeros((1, vector.shape[1]))))
            return vector

        df["features"] = df[columns].values.tolist()
        df["features"] = df.features.apply(lambda x: stack_vectors(x))
        df.drop(columns_to_delete, axis=1, inplace=True)
        vectors = df.values.tolist()

        global_vector = np.zeros(
            (len(vectors), 100, vectors[0][0].shape[1])
        )  # (1500, 100, 768)
        first_vector = pad_vector(vectors[0][0])

        global_vector[0] = first_vector
        for i in range(1, len(vectors) - 1):
            global_vector[i] = pad_vector(vectors[i][0])
        return global_vector

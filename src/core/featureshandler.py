#!/usr/bin/env python

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

from core.embeddinghandler import Embedding, WordEmbedding, TransformerEmbedding
from logger.logger import Logger


class FeaturesHandler:

    _instance = None
    _logger = Logger.instance()

    _transformers: list = [
        "bert-base-multilingual-cased",
        "dccuchile/bert-base-spanish-wwm-cased",
        "PlanTL-GOB-ES/roberta-base-biomedical-es",
        "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
        "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
        "data\\scenario2-taskA\\models\\40epoch\\bert-base-multilingual-cased",
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
            "pos_tag",
        ]
        + _transformers,
        "scenario3-taskB": ["with_entities", "word_emb"] + _transformers,
    }

    @staticmethod
    def instance():
        if FeaturesHandler._instance is None:
            FeaturesHandler()
        return FeaturesHandler._instance

    def __init__(self) -> None:
        from utils.appconstants import AppConstants

        if FeaturesHandler._instance is not None:
            return

        self._task: str = AppConstants.instance()._task

        self._le: LabelEncoder = LabelEncoder()
        self._cv: CountVectorizer = CountVectorizer()
        self._tf: TfidfVectorizer = TfidfVectorizer()
        self._ck: Tokenizer = Tokenizer(char_level=True)
        self._tk: Tokenizer = Tokenizer()

        self._features: list = AppConstants.instance()._features
        self._is_transformer: bool = self.is_transformer()

        """
            0 ['Action']
            1 ['Concept']
            2 ['O']
            3 ['Predicate']
            4 ['Reference']
        """
        self._le.fit(["O", "Action", "Concept", "Predicate", "Reference"])

        FeaturesHandler._instance = self

    def is_transformer(self) -> None:
        return list(filter(lambda x: x in self._features, self._transformers)) != []

    def check_features_for_task(self) -> None:
        for feat in self._features:
            if feat not in self._features_for_task[self._task]:
                import sys

                self._logger.error(f"Feature {feat} don't match with task {self._task}")
                self._logger.error(
                    f"Available features for task {self._task}: {self._features_for_task[self._task]}"
                )
                sys.exit()

    def handle_features(self, df: pd.DataFrame, test: bool = False) -> np.ndarray:
        columns = []
        self._logger.info(
            "Handling features: "
            + ", ".join(self._features)
            + f" for task {self._task}"
        )
        self.check_features_for_task()

        # Both tasks
        if self._is_transformer:
            self._feat_transformer(df)
            columns += ["token"] if "taskA" in self._task else ["token1"] + ["token2"]

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
        if "pos_tag" in self._features:
            self._feat_bag_of_words(df, test=test, column="pos_tag")
            columns += ["pos_tag"]

        # RE fearures
        if "with_entities" in self._features:
            self._feat_with_tags(df, test)
            columns += ["tag1"] + ["tag2"]
        if "word_emb" in self._features:
            self._feat_word_emb(df)
            columns += ["token1"] + ["token2"]

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
        df["tag1"] = self._le.transform(df.tag1.values)
        df["tag2"] = self._le.transform(df.tag2.values)

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

    def fill_token(
        self,
        df: pd.DataFrame,
        original_token: str,
        sentence: str,
        column_int: str,
        tokenized_sent: list,
        vectorized_sent: list,
    ) -> None:
        mask = (df.sentence == sentence) & (
            df["original_token" + column_int] == original_token
        )
        if len(df.loc[mask, "token" + column_int].values) == 0:
            return
        if np.all((np.array(df.loc[mask, "token" + column_int].values[0]) != 0)):
            return
        token = TransformerEmbedding.instance().entity_vector_from_sent(
            original_token, tokenized_sent, vectorized_sent
        )
        df.loc[mask, "token" + column_int] = [np.array(token, dtype="float32")] * len(
            df.loc[mask]
        )

    def _fit_transformer(self, df: pd.DataFrame, column: str) -> None:
        instance = TransformerEmbedding.instance()
        df["token"] = [np.zeros((768,), dtype="float32")] * len(df)

        i: int = 1
        for sent in df.sentence.unique():
            sent_df = df.loc[df.sentence == sent]

            vectorized_sent = instance.sentence_vector(sent)
            tokenized_sent = instance.tokenize(sent)

            for original_token in sent_df.original_token.unique():
                self.fill_token(
                    df, original_token, sent, "", tokenized_sent, vectorized_sent
                )

            self._logger.info(
                f"Finished with sentence {i} of {len(df.sentence.unique())}"
            )
            i += 1

    def _fit_transformer_RE(self, df: pd.DataFrame) -> None:
        instance = TransformerEmbedding.instance()
        df["token1"] = [np.zeros((768,), dtype="float32")] * len(df)
        df["token2"] = [np.zeros((768,), dtype="float32")] * len(df)

        i: int = 1
        for sent in df.sentence.unique():
            sent_df = df.loc[df.sentence == sent]

            vectorized_sent = instance.sentence_vector(sent)
            tokenized_sent = instance.tokenize(sent)

            for original_token1 in sent_df.original_token1.unique():
                self.fill_token(
                    df, original_token1, sent, "1", tokenized_sent, vectorized_sent
                )
            for original_token2 in sent_df.original_token2.unique():
                self.fill_token(
                    df, original_token2, sent, "2", tokenized_sent, vectorized_sent
                )

            self._logger.info(
                f"Finished with sentence {i} of {len(df.sentence.unique())}"
            )
            i += 1

    def _feat_transformer(self, df: pd.DataFrame) -> None:
        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(self._features[0])

        if "taskA" in self._task:
            return self._fit_transformer(df, "token")
        self._fit_transformer_RE(df)

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

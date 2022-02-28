#!/usr/bin/env python

import numpy as np
import pandas as pd
from embeddinghandler import Embedding, WordEmbedding
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from keras.preprocessing.text import Tokenizer


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
        self._cv: CountVectorizer = CountVectorizer()
        self._tf: TfidfVectorizer = TfidfVectorizer()
        self._tk: Tokenizer = Tokenizer(char_level=True)

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
        columns = []

        if "single_word_emb" in self._features:
            FeaturesHandler._feat_single_word_emb(df)
            columns += ["word"]
        if "tf_idf" in self._features:
            FeaturesHandler._feat_tf_idf(df, test=test)
            columns += ["word"]
        if "with_entities" in self._features:
            FeaturesHandler._feat_with_tags(df, test)
            columns += ["tag1"] + ["tag2"]
        if "word_emb" in self._features:
            FeaturesHandler._feat_word_emb(df)
            columns += ["word1"] + ["word2"]
        if "sent_emb" in self._features:
            FeaturesHandler._feat_sent_emb(df)
            columns += ["sentence"]
        if "bag_of_words" in self._features:
            FeaturesHandler._feat_bag_of_words(df, test=test)
            columns += ["word"]
        if "chars" in self._features:
            FeaturesHandler._feat_chars(df, test=test)
            columns += ["word"]

        features: np.ndarray = FeaturesHandler._combine_features(df, columns)
        print("Features matrix succesfully generated, with data shape:", features.shape)
        return np.nan_to_num(features)

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
    def _feat_word_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            tokens = Embedding.prepare_text_to_train(df)
            WordEmbedding.instance().load_model(
                "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4.bin"
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
                "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4.bin"
            )
            WordEmbedding.instance().train_word_emebdding(tokens)
        #     sentences = Embedding.prepare_text_to_train(df)
        #     SentenceEmbedding.instance().train_sentence_emebdding(sentences)
        # df["sentence"] = df.sentence.apply(
        #     lambda x: SentenceEmbedding.instance().sentence_to_vector(x)
        # )
        df["sentence"] = df.sentence.apply(
            lambda x: WordEmbedding.instance().words_to_vector(x.split())
        )

    @staticmethod
    def _feat_bag_of_words(
        df: pd.DataFrame, column: str = "word", test: bool = False
    ) -> None:
        if not test:
            print("Fitting words to bag of words")
            FeaturesHandler.instance()._cv.fit(df[column].unique().tolist())
            print(
                "Vocab size: ", len(FeaturesHandler.instance()._cv.vocabulary_.keys())
            )

        df[column] = df[column].apply(
            lambda x: FeaturesHandler.instance()
            ._cv.transform([x])
            .toarray()
            .reshape(-1)
        )

    @staticmethod
    def _feat_single_word_emb(df: pd.DataFrame) -> None:
        if not Embedding.trained():
            WordEmbedding.instance().load_model(
                "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4.bin"
            )

        df["word"] = df.word.apply(lambda x: WordEmbedding.instance().word_vector(x))

    @staticmethod
    def _feat_tf_idf(
        df: pd.DataFrame, column: str = "word", test: bool = False
    ) -> None:
        if not test:
            print("Fitting words to tf idf")
            FeaturesHandler.instance()._tf.fit(df[column].unique().tolist())
            print(
                "Vocab size: ", len(FeaturesHandler.instance()._tf.vocabulary_.keys())
            )

        df[column] = df[column].apply(
            lambda x: FeaturesHandler.instance()
            ._tf.transform([x])
            .toarray()
            .reshape(-1)
        )

    @staticmethod
    def _feat_chars(df: pd.DataFrame, test: bool = False) -> None:
        from keras.preprocessing.sequence import pad_sequences

        if not test:
            FeaturesHandler.instance()._tk.fit_on_texts(df.word)
            vocab_size = len(FeaturesHandler.instance()._tk.word_index) + 1
            print(f"Vocab size: {vocab_size}")

        df["word"] = df.word.apply(
            lambda x: FeaturesHandler.instance()._tk.texts_to_sequences([x])[0]
        )
        df["word"] = df.word.apply(lambda x: pad_sequences([x], maxlen=15)[0])
        print(f"Matriz features for emebdding: {df.word.shape}")

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

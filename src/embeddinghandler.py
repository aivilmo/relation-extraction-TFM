#!/usr/bin/env python
from gensim.models.doc2vec import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text


class Embedding:
    @staticmethod
    def prepare_text_to_train(df: pd.DataFrame) -> list:
        sentences = pd.DataFrame(data={"unique_sentences": df.sentence.unique()})
        return list(sentences.unique_sentences.apply(lambda x: x.split()))

    @staticmethod
    def trained() -> bool:
        return (
            WordEmbedding.instance()._keyed_vectors != None
            or BERTEmbedding.instance()._preprocess_layer != None
        )


class BERTEmbedding(Embedding):
    _instance = None

    @staticmethod
    def instance():
        if BERTEmbedding._instance == None:
            BERTEmbedding()
        return BERTEmbedding._instance

    def __init__(self) -> None:
        if BERTEmbedding._instance != None:
            raise Exception

        self._preprocess_layer = None
        self._encoder_layer = None
        BERTEmbedding._instance = self

    def build_BERT_model(self) -> None:
        from transformers import BertTokenizer, BertModel

        self._preprocess_layer = BertTokenizer.from_pretrained("bert-base-uncased")
        self._encoder_layer = BertModel.from_pretrained("bert-base-uncased")

        print("Building BERT model with preprocessor: bert-base-uncased")
        print("Building BERT model from: bert-base-uncased")

    def build_distilBERT_model(self) -> None:
        from transformers import DistilBertTokenizer, DistilBertModel

        self._preprocess_layer = DistilBertTokenizer.from_pretrained(
            "distilbert-base-uncased"
        )
        self._encoder_layer = DistilBertModel.from_pretrained("distilbert-base-uncased")

        print("Building distilBERT model with preprocessor: distilbert-base-uncased")
        print("Building distilBERT model from: distilbert-base-uncased")

    def word_vector(self, word: str) -> np.ndarray:
        word_preprocessed = self._preprocess_layer(word, return_tensors="pt")
        bert_results = self._encoder_layer(**word_preprocessed)
        return bert_results.last_hidden_state.detach().numpy()[0][0]

    def plot_model(self) -> None:
        filename = "images\\bert.png"
        tf.keras.utils.plot_model(self._model, to_file=filename)
        print(f"Ploted model to file: {filename}")


class WordEmbedding(Embedding):
    _instance = None

    @staticmethod
    def instance():
        if WordEmbedding._instance == None:
            WordEmbedding()
        return WordEmbedding._instance

    def __init__(self) -> None:
        if WordEmbedding._instance != None:
            raise Exception

        self._model = None
        self._keyed_vectors = None
        WordEmbedding._instance = self

    def load_model(self, filename: str) -> None:
        print(f"Loading keyed vectors from: {filename}")

        self._keyed_vectors = KeyedVectors.load_word2vec_format(filename, binary=True)

    def train_word_emebdding(self, tokens: list) -> None:
        print("Training model word embedding...")

        self._model = Word2Vec(
            sentences=tokens,
            vector_size=100,
            window=5,
            min_count=0,
            workers=-1,
            sg=0,
        )
        self._model.wv = self._keyed_vectors
        self._model.build_vocab(tokens, update=True)
        self._model.train(
            tokens, total_examples=self._model.corpus_count, epochs=self._model.epochs
        )
        self._keyed_vectors = self._model.wv

    def words_to_vector(self, words: list) -> np.ndarray:
        word_matrix: np.ndarray = np.zeros((len(words), 100))
        for w in range(len(words)):
            if words[w] in list(self._keyed_vectors.index_to_key):
                word_matrix[w] = self._keyed_vectors.get_vector(words[w])
        return np.mean(word_matrix, axis=0)

    def word_vector(self, word: str) -> np.ndarray:
        if word in list(self._keyed_vectors.index_to_key):
            return self._keyed_vectors.get_vector(word)
        return np.zeros((100,))

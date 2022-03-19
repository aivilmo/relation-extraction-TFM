#!/usr/bin/env python
from gensim.models.doc2vec import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text

from logger import Logger


class Embedding:
    @staticmethod
    def prepare_text_to_train(df: pd.DataFrame) -> list:
        sentences = pd.DataFrame(data={"unique_sentences": df.sentence.unique()})
        return list(sentences.unique_sentences.apply(lambda x: x.split()))

    @staticmethod
    def trained() -> bool:
        return (
            WordEmbedding.instance()._keyed_vectors != None
            or TransformerEmbedding.instance()._preprocess_layer != None
        )


class TransformerEmbedding(Embedding):
    _instance = None

    @staticmethod
    def instance():
        if TransformerEmbedding._instance == None:
            TransformerEmbedding()
        return TransformerEmbedding._instance

    def __init__(self) -> None:
        if TransformerEmbedding._instance != None:
            raise Exception

        self._preprocess_layer = None
        self._encoder_layer = None
        self._logger = Logger.instance()
        TransformerEmbedding._instance = self

    def build_transformer(self, type: str = "bert-base-multilingual-cased") -> None:
        self._logger.info(f"Building transformer model with preprocessor: {type}")
        self._logger.info(f"Building transformer model from: {type}")

        if "distil" in type:
            from transformers import DistilBertTokenizer, DistilBertModel

            self._preprocess_layer = DistilBertTokenizer.from_pretrained(type)
            self._encoder_layer = DistilBertModel.from_pretrained(type)
        elif "bert-base" in type:
            from transformers import BertTokenizer, BertModel

            self._preprocess_layer = BertTokenizer.from_pretrained(type)
            self._encoder_layer = BertModel.from_pretrained(type)
        elif type == "gpt2":
            from transformers import GPT2Tokenizer, GPT2Model

            self._preprocess_layer = GPT2Tokenizer.from_pretrained(type)
            self._encoder_layer = GPT2Model.from_pretrained(type)
        elif "roberta" in type:
            from transformers import AutoTokenizer, AutoModel

            self._preprocess_layer = AutoTokenizer.from_pretrained(type)
            self._encoder_layer = AutoModel.from_pretrained(type)

        self._logger.info(f"Transformer has built with preprocessor: {type}")
        self._logger.info(f"Transformer has built from: {type}")

    def word_vector(self, word: str) -> np.ndarray:
        word_preprocessed = self._preprocess_layer(word, return_tensors="pt")
        bert_results = self._encoder_layer(**word_preprocessed)
        return bert_results.last_hidden_state.detach().numpy()[0][0]

    def sentence_vector(self, sentence: str) -> np.ndarray:
        sentence_preprocessed = self._preprocess_layer(sentence, return_tensors="pt")
        bert_results = self._encoder_layer(**sentence_preprocessed)
        return bert_results.last_hidden_state.detach().numpy()[0]

    def tokenize(self, text: str) -> list:
        return self._preprocess_layer.tokenize(text)

    def plot_model(self) -> None:
        filename = "images\\bert.png"
        tf.keras.utils.plot_model(self._model, to_file=filename)
        self._logger.info(f"Ploted model to file: {filename}")


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
        self._logger = Logger.instance()
        WordEmbedding._instance = self

    def load_model(self, filename: str) -> None:
        self._logger.info(f"Loading keyed vectors from: {filename}")

        self._keyed_vectors = KeyedVectors.load_word2vec_format(filename, binary=True)

    def train_word_emebdding(self, tokens: list) -> None:
        self._logger.info("Training model word embedding...")

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

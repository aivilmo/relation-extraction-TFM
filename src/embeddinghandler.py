#!/usr/bin/env python
from typing import Iterable
from gensim.models.doc2vec import Doc2Vec, Word2Vec, TaggedDocument, KeyedVectors
import numpy as np
import pandas as pd


class Embedding:
    @staticmethod
    def prepare_text_to_train(df: pd.DataFrame) -> list:
        sentences = pd.DataFrame(data={"unique_sentences": df.sentence.unique()})
        return list(sentences.unique_sentences.apply(lambda x: x.split()))


class SentenceEmbedding(Embedding):
    _instance = None

    @staticmethod
    def instance():
        if SentenceEmbedding._instance == None:
            SentenceEmbedding()
        return SentenceEmbedding._instance

    def __init__(self) -> None:
        if SentenceEmbedding._instance != None:
            raise Exception

        self._model = None
        SentenceEmbedding._instance = self

    def train_sentence_emebdding(self, tokenized_sentences: list) -> None:
        print("Training model sentences embedding...")

        def tagged_document(sentences_list: list) -> Iterable:
            for i, list_of_words in enumerate(sentences_list):
                yield TaggedDocument(list_of_words, [i])

        tagged_docs: list = list(tagged_document(tokenized_sentences))
        self._model = Doc2Vec(vector_size=40, min_count=1, epochs=30, workers=-1)
        self._model.build_vocab(tagged_docs)
        self._model.train(
            tagged_docs,
            total_examples=self._model.corpus_count,
            epochs=self._model.epochs,
        )

    def sentence_to_vector(self, sentence: str) -> np.ndarray:
        from preprocess import Preprocessor

        sent_tokens = Preprocessor.preprocess(sentence).split()
        return self._model.infer_vector(sent_tokens)


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
        print("After train ", len(self._model.wv.index_to_key))

    def words_to_vector(self, words: list) -> np.ndarray:
        word_matrix: np.ndarray = np.zeros((len(words), 100))
        for w in range(len(words)):
            if words[w] in list(self._keyed_vectors.index_to_key):
                word_matrix[w] = self._keyed_vectors.get_vector(words[w])
        return np.mean(word_matrix, axis=0)

    def word_vector(self, word: str) -> np.ndarray:
        return self._keyed_vectors.get_vector(word)

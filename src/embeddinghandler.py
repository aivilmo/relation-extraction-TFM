#!/usr/bin/env python
from typing import Iterable
from gensim.models.doc2vec import Doc2Vec, Word2Vec, TaggedDocument
import numpy as np
import pandas as pd


class Embedding:
    @staticmethod
    def prepare_text_to_train(df: pd.DataFrame) -> list:
        sentences = pd.DataFrame(data={"unique_sentences": df.sentence.unique()})
        return list(sentences.unique_sentences.apply(lambda x: x.split()))


class SentenceEmbedding(Embedding):
    @staticmethod
    def train_sentence_emebdding(tokenized_sentences: list) -> Doc2Vec:
        def tagged_document(sentences_list: list) -> Iterable:
            for i, list_of_words in enumerate(sentences_list):
                yield TaggedDocument(list_of_words, [i])

        tagged_docs: list = list(tagged_document(tokenized_sentences))
        model = Doc2Vec(vector_size=40, min_count=1, epochs=30, workers=-1)
        model.build_vocab(tagged_docs)
        model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    @staticmethod
    def sentence_to_vector(model: Doc2Vec, sentence: str) -> np.ndarray:
        from preprocess import Preprocessor

        sent_tokens = Preprocessor.preprocess(sentence).split()
        return model.infer_vector(sent_tokens)


class WordEmbedding(Embedding):
    @staticmethod
    def train_word_emebdding(tokens: list) -> Word2Vec:
        model = Word2Vec(
            sentences=tokens,
            vector_size=100,
            window=5,
            min_count=2,
            workers=-1,
            sg=0,
        )

        model.build_vocab(tokens)
        model.train(tokens, total_examples=model.corpus_count, epochs=model.epochs)
        return model

    @staticmethod
    def words_to_vector(model: Word2Vec, words: list) -> np.ndarray:
        word_matrix: np.ndarray = np.zeros((len(words), 100))
        for w in range(len(words)):
            if words[w] in list(model.wv.index_to_key):
                word_matrix[w] = model.wv.word_vec(words[w])
        return np.mean(word_matrix, axis=0)

    @staticmethod
    def word_vector(model: Word2Vec, word: str) -> np.ndarray:
        return model.wv.word_vec(word)

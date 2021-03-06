#!/usr/bin/env python
from gensim.models.doc2vec import Word2Vec, KeyedVectors
import numpy as np
import pandas as pd
import tensorflow as tf

from logger.logger import Logger


class Embedding:
    _logger = Logger.instance()

    @staticmethod
    def prepare_text_to_train(df: pd.DataFrame) -> list:
        sentences = pd.DataFrame(data={"unique_sentences": df.sentence.unique()})
        return list(sentences.unique_sentences.apply(lambda x: x.split()))

    @staticmethod
    def trained() -> bool:
        return (
            WordEmbedding.instance()._keyed_vectors is not None
            or TransformerEmbedding.instance()._preprocess_layer is not None
        )


class TransformerEmbedding(Embedding):

    _instance = None

    @staticmethod
    def instance():
        if TransformerEmbedding._instance is None:
            TransformerEmbedding()
        return TransformerEmbedding._instance

    def __init__(self) -> None:
        if TransformerEmbedding._instance is not None:
            raise Exception

        self._preprocess_layer = None
        self._encoder_layer = None
        TransformerEmbedding._instance = self

    def build_transformer(self, type: str) -> None:
        from transformers import AutoTokenizer, AutoModel

        if type == "":
            type = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"

        Embedding._logger.info(f"Building transformer model with preprocessor: {type}")
        Embedding._logger.info(f"Building transformer model from: {type}")

        self._preprocess_layer = AutoTokenizer.from_pretrained(type)
        self._encoder_layer = AutoModel.from_pretrained(type)

        Embedding._logger.info(f"Transformer has built with preprocessor: {type}")
        Embedding._logger.info(f"Transformer has built from: {type}")

    def build_transformer_to_finetuning(self, type: str, classes: int) -> None:
        from transformers import AutoTokenizer, AutoModelForTokenClassification

        if type == "":
            type = "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"

        Embedding._logger.info(f"Building transformer model with preprocessor: {type}")
        Embedding._logger.info(f"Building transformer model to finetuning from: {type}")

        self._preprocess_layer = AutoTokenizer.from_pretrained(type)
        self._encoder_layer = AutoModelForTokenClassification.from_pretrained(
            type, num_labels=classes
        )

        Embedding._logger.info(f"Transformer has built with preprocessor: {type}")
        Embedding._logger.info(f"Transformer to finetuning has built from: {type}")

    def word_vector(self, word: str) -> np.ndarray:
        word_preprocessed = self._preprocess_layer(word, return_tensors="pt")
        bert_results = self._encoder_layer(**word_preprocessed)
        return bert_results.last_hidden_state.detach().numpy()[0][0]

    def sentence_vector(self, sentence: str) -> np.ndarray:
        sentence_preprocessed = self._preprocess_layer(sentence, return_tensors="pt")
        bert_results = self._encoder_layer(**sentence_preprocessed)
        return bert_results.last_hidden_state.detach().numpy()[0]

    def tokenize_input_ids(self, sentence: str) -> np.ndarray:
        sentence_preprocessed = self._preprocess_layer(sentence, return_tensors="pt")
        return sentence_preprocessed.input_ids.detach().numpy()[0]

    def tokenize(self, text: str) -> list:
        return self._preprocess_layer.tokenize(text)

    def apply_transformer(self, token: str, sentence: str) -> np.ndarray:
        sent = self.sentence_vector(sentence)
        tokenized_sent = self.tokenize(sentence)
        tokens = self.tokenize(token)

        token_vectors = [
            sent[i + 1] for i, e in enumerate(tokenized_sent) if e in set(tokens)
        ]
        vector = np.array(token_vectors)

        if vector.shape == (0,):
            return np.zeros((768,))
        return np.mean(vector, axis=0)

    def entity_vector_from_sent(
        self, token: str, tokenized_sent: list, vectorized_sent: list
    ) -> np.ndarray:
        tokens = self.tokenize(token)
        token_vectors = [
            vectorized_sent[i + 1]
            for i, e in enumerate(tokenized_sent)
            if e in set(tokens)
        ]
        vector = np.array(token_vectors)

        if vector.shape == (0,):
            return np.zeros((768,))
        return np.mean(vector, axis=0)

    def plot_model(self) -> None:
        filename = "images\\bert.png"
        tf.keras.utils.plot_model(self._model, to_file=filename)
        Embedding._logger.info(f"Ploted model to file: {filename}")


class WordEmbedding(Embedding):

    _instance = None
    _word2vec_file = "..\\dataset\\word-embeddings_fasttext\\EMEA+scielo-es_skipgram_w=10_dim=100_minfreq=1_neg=10_lr=1e-4.bin"

    @staticmethod
    def instance():
        if WordEmbedding._instance is None:
            WordEmbedding()
        return WordEmbedding._instance

    def __init__(self) -> None:
        if WordEmbedding._instance is not None:
            raise Exception

        self._model = None
        self._keyed_vectors = None

        WordEmbedding._instance = self

    def load_model(self) -> None:
        Embedding._logger.info(
            f"Loading keyed vectors from: {WordEmbedding._word2vec_file}"
        )

        self._keyed_vectors = KeyedVectors.load_word2vec_format(
            WordEmbedding._word2vec_file, binary=True
        )

    def train_word_embedding(self, tokens: list) -> None:
        Embedding._logger.info("Training model word embedding...")

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

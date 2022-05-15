#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings
from abc import abstractmethod

from logger.logger import Logger

warnings.simplefilter(action="ignore", category=FutureWarning)


class Preprocessor:

    _instance = None
    _n_classes = None

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if Preprocessor._instance is None:
            Preprocessor()
        return Preprocessor._instance

    def __init__(self) -> None:
        if Preprocessor._instance is not None:
            raise Exception

        self._le = LabelEncoder()

        Preprocessor._instance = self

    @abstractmethod
    def process_content(self, path: Path) -> pd.DataFrame:
        pass

    @abstractmethod
    def process_content_cased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        pass

    @abstractmethod
    def process_content_uncased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        pass

    def train_test_split(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:
        from core.featureshandler import FeaturesHandler

        # Transform labels
        y_train, y_test = self.encode_labels(train_df, test_df, y_column=y_column)

        train_df.drop(y_column, axis=1, inplace=True)
        test_df.drop(y_column, axis=1, inplace=True)

        # Handle features from data
        X_train = FeaturesHandler.instance().handle_features(train_df)
        X_test = FeaturesHandler.instance().handle_features(test_df, test=True)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def data_iterator(
        df: pd.DataFrame, chunk_size: int = 1500, y_column: str = "tag"
    ) -> np.ndarray:
        from core.featureshandler import FeaturesHandler
        import dask.dataframe as dd
        from dask import delayed, compute

        df.drop(y_column, axis=1, inplace=True)

        delayed_results = []
        dask_df = dd.from_pandas(df, chunksize=chunk_size)
        partitions = len(dask_df.to_delayed())
        part_n = 1
        for part in dask_df.to_delayed():
            df = part.compute()
            Preprocessor._logger.info(
                f"Computing part of dataframe, part {part_n} of {partitions}"
            )
            part_n += 1
            X = delayed(FeaturesHandler.instance().handle_features)(df)
            delayed_results.append(X)

        start = time.time()
        results = compute(*delayed_results, scheduler="threads")
        Preprocessor._logger.info(
            f"Parallel features handler: {(time.time() - start) / 60} minutes"
        )
        return results

    @staticmethod
    def preprocess(text: str, without_stopwords: bool = False) -> str:
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk.corpus import stopwords
        import re
        import unidecode

        unaccented_string = unidecode.unidecode(text)
        alphanumeric_text = re.sub("[^0-9a-zA-Z-/:]+", " ", unaccented_string)
        without_accents = (
            alphanumeric_text.replace("á", "a")
            .replace("é", "e")
            .replace("í", "i")
            .replace("ó", "o")
            .replace("ú", "u")
        )
        tokens = without_accents.lower().split()

        if without_stopwords:
            stop_words = set(stopwords.words("spanish"))
            tokens = [token for token in tokens if token not in stop_words]
            return " ".join(tokens)

        wordnet_lemmatizer = WordNetLemmatizer()
        tokens_lemmas = [wordnet_lemmatizer.lemmatize(token) for token in tokens]

        stemmer = PorterStemmer()
        tokens_stemmed = [stemmer.stem(token) for token in tokens_lemmas]

        return " ".join(tokens_stemmed)

    @staticmethod
    def prepare_labels(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        from keras.utils.np_utils import to_categorical

        return to_categorical(
            y_train, num_classes=Preprocessor._n_classes
        ), to_categorical(y_test, num_classes=Preprocessor._n_classes)

    @staticmethod
    def prepare_2D_labels(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        from keras.preprocessing.sequence import pad_sequences

        y_train, y_test = pad_sequences(
            y_train.values, maxlen=100, padding="post", value=8
        ), pad_sequences(y_test.values, maxlen=100, padding="post", value=8)

        y_train = y_train.reshape((y_train.shape[0], y_train.shape[1], 1))
        y_test = y_test.reshape((y_test.shape[0], y_test.shape[1], 1))

        return y_train, y_test

    def encode_labels(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:
        from model.coremodel import AbstractModel

        self._logger.info(f"Transforming {y_column} into labels")
        y_train = self._le.fit_transform(train_df[y_column].values)
        y_test = self._le.transform(test_df[y_column].values)

        AbstractModel.set_labels(list(self._le.classes_))
        Preprocessor._n_classes = len(list(self._le.classes_))

        return y_train, y_test

    def encode_2D_labels(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:

        from model.coremodel import AbstractModel

        self._logger.info(f"Transforming {y_column} into 2D labels")
        y_train = train_df[y_column].apply(self._le.transform)
        y_test = test_df[y_column].apply(self._le.transform)

        AbstractModel.instance().set_labels(list(self._le.classes_))
        self._n_classes = len(list(self._le.classes_))

        return y_train, y_test

    @staticmethod
    def remove_invalid_classes(
        df_to_remove: pd.DataFrame, df_from_remove: pd.DataFrame, y_column: str
    ) -> None:
        invalid_labels: list = list(
            set(df_to_remove[y_column].unique())
            - set(df_from_remove[y_column].unique())
        )

        df_to_remove.drop(
            df_to_remove.loc[df_to_remove[y_column].isin(invalid_labels)].index,
            inplace=True,
        )

    @staticmethod
    def data_augmentation(
        df: pd.DataFrame,
        transformer_type: str,
        last_n_classes: int = 3,
        classes_to_augment: list = [],
    ) -> pd.DataFrame:
        def synsets(word: str) -> list:
            from nltk.corpus import wordnet as wn
            from googletrans import Translator

            translator = Translator()
            translation: str = translator.translate(word, src="es", dest="en").text
            synsets: list = []
            for synset in wn.synsets(translation):
                synsets += synset.lemma_names()
                synsets += synset.lemma_names("spa")
                synsets += synset.lemma_names("ita")
                synsets += synset.lemma_names("fra")
            return synsets

        from core.embeddinghandler import Embedding, TransformerEmbedding

        Preprocessor._logger.info(
            f"Starting data augmentation for the {last_n_classes}th least represented classes"
        )

        augmented_df: pd.DataFrame = df.copy()
        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        tags = df.tag.value_counts().index.tolist()[-last_n_classes:]
        if classes_to_augment:
            tags = classes_to_augment

        Preprocessor._logger.info(
            "Data augmentation for classes: " + ", ".join(classes_to_augment)
        )

        index: int = df.iloc[-1].name
        for tag in tags:
            Preprocessor._logger.info(f"Data augmentation for class {tag}")
            for value in df.query("tag=='" + tag + "'")["token"].values:
                sentences = df[(df.tag == tag) & (df.token == value)]["sentence"].values
                for sent in sentences:
                    tokenized_sent = TransformerEmbedding.instance().tokenize(sent)
                    for syn in synsets(value):
                        synset = TransformerEmbedding.instance().tokenize(syn)[0]
                        new_tokenized_sent = [
                            synset if token == value else token
                            for token in tokenized_sent
                        ]
                        vector = TransformerEmbedding.instance().sentence_vector(
                            " ".join(new_tokenized_sent)
                        )
                        position = new_tokenized_sent.index(synset)
                        word = pd.Series(
                            {
                                "token": synset,
                                "vector": vector[position + 1],
                                "tag": tag,
                                "sentence": sent,
                            },
                            name=index,
                        )
                        index += 1
                        augmented_df = augmented_df.append(word)

        Preprocessor._logger.info(
            f"Finished data augmentation, added {index - df.iloc[-1].name} new synsets"
        )
        return augmented_df


class NERPreprocessor(Preprocessor):

    _instance = None

    @staticmethod
    def instance():
        if NERPreprocessor._instance is None:
            NERPreprocessor()
        return NERPreprocessor._instance

    def __init__(self) -> None:
        if NERPreprocessor._instance is not None:
            raise Exception

        NERPreprocessor._instance = self

    def process_content(self, path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        self._logger.info(f"Loaded {len(collection)} sentences for fitting.")
        self._logger.info(f"process_content")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0
        sentence_id: int = 0

        for sentence in collection.sentences:
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = keyphrases.text.split()
                for i in range(len(entities)):
                    tag = "B-" if i == 0 else "I-"
                    old_entity = sentence_entities.get(entities[i], [])
                    old_entity.append(tag + keyphrases.label)
                    sentence_entities[entities[i]] = old_entity

            for word in sentence.text.split():
                tag = sentence_entities.get(word, ["O"])
                word = pd.Series(
                    {
                        "token": word,
                        "original_token": word,
                        "tag": max(set(tag), key=tag.count),
                        "sentence": sentence.text,
                    },
                    name=index,
                )
                index += 1
                df = df.append(word)
            sentence_id += 1

        self._logger.info(f"Training completed: Stored {index} words.")
        return df

    def process_content_cased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        from ehealth.anntools import Collection
        from core.embeddinghandler import Embedding, TransformerEmbedding

        from collections import defaultdict

        collection = Collection().load_dir(path)
        self._logger.info(f"Loaded {len(collection)} sentences for fitting.")
        self._logger.info(f"process_content_cased_transformer")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        for sentence in collection.sentences:
            prep_sent = (
                (Preprocessor.preprocess(sentence.text))
                .replace(".", "")
                .replace(",", "")
                .strip()
            )

            sent = TransformerEmbedding.instance().sentence_vector(prep_sent)
            tokenized_sent = TransformerEmbedding.instance().tokenize(prep_sent)
            sentence_entities = defaultdict(lambda: [])
            if sentence.keyphrases == []:
                continue
            for keyphrase in sentence.keyphrases:
                entities = keyphrase.text.split()
                for i in range(len(entities)):
                    prep_entity = Preprocessor.preprocess(entities[i])
                    entity_word = TransformerEmbedding.instance().tokenize(prep_entity)
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + keyphrase.label
                    for j in range(len(entity_word)):
                        if j != 0:
                            break
                        sentence_entities[entity_word[j]].append(tag)

            token_pos: int = 0
            original_sent: list = (
                sentence.text.replace(".", " ").replace(",", " ").split()
            )

            for i in range(len(tokenized_sent)):
                token: str = tokenized_sent[i]
                if token.startswith("Ġ"):
                    original_token: str = original_sent[token_pos]
                    token_pos += 1
                if sentence_entities[token] == []:
                    tag = "O"
                else:
                    tag = sentence_entities[token].pop(0)
                word = pd.Series(
                    {
                        "token": token,
                        "original_token": original_token,
                        "vector": sent[i + 1],
                        "tag": tag,
                        "sentence": sentence.text,
                    },
                    name=index,
                )
                index += 1
                df = df.append(word)

        self._logger.info(f"Training completed: Stored {index} words.")
        return df

    def process_content_uncased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        print("NERPreprocessor process_content_uncased_transformer")


class REPreprocessor(Preprocessor):

    _instance = None

    @staticmethod
    def instance():
        if REPreprocessor._instance is None:
            REPreprocessor()
        return REPreprocessor._instance

    def __init__(self) -> None:
        if REPreprocessor._instance is not None:
            raise Exception

        REPreprocessor._instance = self

    def process_content(self, path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        self._logger.info(f"Loaded {len(collection)} sentences for fitting.")
        self._logger.info(f"process_content")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0
        sent_id: int = 1

        for sentence in collection.sentences:
            relation_pairs = {}
            sentence_entities = {}
            if sentence.relations == []:
                continue
            for relation in sentence.relations:
                from_relation = relation.from_phrase.text.split()
                for i in range(len(from_relation)):
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + relation.from_phrase.label
                    if i == 0:
                        from_word = from_relation[i]
                    from_entity = tag
                    sentence_entities[from_relation[i]] = (
                        tag,
                        relation.from_phrase.text,
                    )

                to_relation = relation.to_phrase.text.split()
                for i in range(len(to_relation)):
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + relation.to_phrase.label
                    if i == 0:
                        to_word = to_relation[i]
                    to_entity = tag
                    sentence_entities[to_relation[i]] = (tag, relation.to_phrase.text)

                relation_pairs[
                    (from_word, from_entity, to_word, to_entity)
                ] = relation.label

            for from_word in sentence.text.split():
                for to_word in sentence.text.split():
                    from_entity, original_token1 = sentence_entities.get(
                        from_word, ("O", from_word)
                    )
                    to_entity, original_token2 = sentence_entities.get(
                        to_word, ("O", to_word)
                    )
                    relation = relation_pairs.get(
                        (from_word, from_entity, to_word, to_entity), "O"
                    )
                    relation = pd.Series(
                        {
                            "token1": from_word,
                            "original_token1": original_token1,
                            "tag1": from_entity,
                            "token2": to_word,
                            "original_token2": original_token2,
                            "tag2": to_entity,
                            "relation": relation,
                            "sentence": sentence.text,
                        },
                        name=index,
                    )
                    index += 1
                    df = df.append(relation)
            sent_id += 1
            self._logger.info(f"Finished sentence {sent_id} of {len(collection)}")

        self._logger.info(f"Training completed: Stored {index} word pairs.")
        return df

    def process_content_cased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        from ehealth.anntools import Collection
        from core.embeddinghandler import Embedding, TransformerEmbedding

        collection = Collection().load_dir(path)
        self._logger.info(f"Loaded {len(collection)} sentences for fitting.")
        self._logger.info(f"process_content_cased_transformer")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0
        sent_id: int = 1
        sentences = []

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        for sentence in collection.sentences:
            prep_sent = (
                (Preprocessor.preprocess(sentence.text))
                .replace(".", "")
                .replace(",", "")
                .strip()
            )

            if sentence.text in sentences or sentence.relations == []:
                continue
            sentences.append(sentence.text)

            sent = TransformerEmbedding.instance().sentence_vector(prep_sent)
            tokenized_sent = TransformerEmbedding.instance().tokenize(prep_sent)
            relation_pairs = {}
            sentence_entities = {}
            for relation in sentence.relations:
                prep_from_relation = Preprocessor.preprocess(relation.from_phrase.text)
                from_relation = TransformerEmbedding.instance().tokenize(
                    prep_from_relation
                )
                for i in range(len(from_relation)):
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + relation.from_phrase.label
                    if i != 0:
                        break
                    from_word = from_relation[i]
                    from_entity = tag
                    sentence_entities[from_word] = (tag, prep_from_relation)

                prep_to_relation = Preprocessor.preprocess(relation.to_phrase.text)
                to_relation = TransformerEmbedding.instance().tokenize(prep_to_relation)
                for i in range(len(to_relation)):
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + relation.to_phrase.label
                    if i != 0:
                        break
                    to_word = to_relation[i]
                    to_entity = tag
                    sentence_entities[to_word] = (tag, prep_to_relation)

                relation_pairs[
                    (from_word, from_entity, to_word, to_entity)
                ] = relation.label

            token1_pos: int = 0
            token2_pos: int = 0
            original_sent: list = (
                sentence.text.replace(".", " ").replace(",", " ").split()
            )
            for i in range(len(tokenized_sent)):
                from_word = tokenized_sent[i]
                if from_word.startswith("Ġ"):
                    original_token1: str = original_sent[token1_pos]
                    token1_pos += 1
                from_entity, original_token1 = sentence_entities.get(
                    from_word, ("O", original_token1)
                )
                for j in range(len(tokenized_sent)):
                    to_word = tokenized_sent[j]
                    if to_word.startswith("Ġ"):
                        original_token2: str = original_sent[token2_pos]
                        token2_pos += 1
                    to_entity, original_token2 = sentence_entities.get(
                        to_word, ("O", original_token2)
                    )
                    relation = relation_pairs.get(
                        (from_word, from_entity, to_word, to_entity), "O"
                    )
                    relation = pd.Series(
                        {
                            "token1": from_word,
                            "original_token1": original_token1,
                            "vector1": sent[i + 1],
                            "tag1": from_entity,
                            "token2": to_word,
                            "original_token2": original_token2,
                            "vector2": sent[j + 1],
                            "tag2": to_entity,
                            "relation": relation,
                            "sentence": sentence.text,
                        },
                        name=index,
                    )
                    index += 1
                    df = df.append(relation)

                token2_pos = 0

            sent_id += 1
            self._logger.info(f"Finished sentence {sent_id} of {len(collection)}")

        self._logger.info(f"Training completed: Stored {index} word pairs.")
        return df

    def process_content_uncased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        print("REPreprocessor process_content_uncased_transformer")

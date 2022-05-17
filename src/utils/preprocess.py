#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings
from abc import abstractmethod

from ehealth.anntools import Collection
from logger.logger import Logger
from core.embeddinghandler import Embedding, TransformerEmbedding

warnings.simplefilter(action="ignore", category=FutureWarning)


class Preprocessor:

    _instance = None
    _n_classes = None
    _default_tag = "O"

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
                tag = sentence_entities.get(word, [self._default_tag])
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
                    tag = self._default_tag
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

    def update_entity(self, phrase, entities) -> dict:
        entities[phrase.text] = phrase.label
        return entities

    def update_transformer_entity(
        self, phrase, sent, sent_vector, entities
    ) -> tuple[str, dict]:
        prep_relation = Preprocessor.preprocess(phrase.text)
        relation = " ".join(TransformerEmbedding.instance().tokenize(prep_relation))

        vector = np.zeros((len(relation.split()), 768))
        i: int = 0
        for w in relation.split():
            index = sent.index(w)
            vector[i] = sent_vector[index]
            i += 1

        entities[relation] = (phrase.label, vector.mean(axis=0))
        return relation, entities

    def append_relations(self, sentence: list, relation: str) -> list:
        if relation not in sentence:
            sentence.append(relation)
        return sentence

    def process_content(self, path: Path) -> pd.DataFrame:
        collection = Collection().load_dir(path)
        self._logger.info(f"Loaded {len(collection)} sentences for fitting.")
        self._logger.info(f"process_content")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0
        sent_id: int = 1

        for sentence in collection.sentences:
            if sentence.relations == []:
                continue

            relation_pairs, from_entities, to_entities = {}, {}, {}
            sentence_ent: list = []
            sent: str = sentence.text

            for relation in sentence.relations:
                relation_from = relation.from_phrase
                from_entities = self.update_entity(relation_from, from_entities)

                relation_to = relation.to_phrase
                to_entities = self.update_entity(relation_to, to_entities)

                relation_pairs[(relation_from.text, relation_to.text)] = relation.label

                sent = sent.replace(relation_from.text, "")
                sent = sent.replace(relation_to.text, "")
                sentence_ent = self.append_relations(sentence_ent, relation_from.text)
                sentence_ent = self.append_relations(sentence_ent, relation_to.text)

            sentence_ent = sent.split() + sentence_ent
            for from_word in sentence_ent:
                from_entity = from_entities.get(from_word, self._default_tag)
                for to_word in sentence_ent:
                    if from_word == to_word:
                        continue
                    to_entity = to_entities.get(to_word, self._default_tag)

                    pair = (from_word, to_word)
                    relation = relation_pairs.get(pair, self._default_tag)
                    relation = pd.Series(
                        {
                            "token1": from_word,
                            "original_token1": from_word,
                            "tag1": from_entity,
                            "token2": to_word,
                            "original_token2": to_word,
                            "tag2": to_entity,
                            "relation": relation,
                            "sentence": sentence.text,
                        },
                        name=index,
                    )
                    index += 1
                    df = df.append(relation)

            self._logger.info(f"Finished sentence {sent_id} of {len(collection)}")
            sent_id += 1

        self._logger.info(f"Training completed: Stored {index} word pairs.")
        return df

    def process_content_cased_transformer(
        self, path: Path, transformer_type: str
    ) -> pd.DataFrame:
        collection = Collection().load_dir(path)
        self._logger.info(f"Loaded {len(collection)} sentences for fitting.")
        self._logger.info(f"process_content_cased_transformer")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0
        sent_id: int = 1
        default_pair: tuple = (self._default_tag, np.zeros((1, 768)))

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        for sentence in collection.sentences:
            if sentence.relations == []:
                continue

            prep_sent = (
                (Preprocessor.preprocess(sentence.text))
                .replace(".", "")
                .replace(",", "")
                .strip()
            )

            sentence_ent, sentence_ori = [], []
            sent = TransformerEmbedding.instance().sentence_vector(prep_sent)
            tokenized_sent = TransformerEmbedding.instance().tokenize(prep_sent)
            sent_joined: str = " ".join(tokenized_sent)
            original_sent: str = sentence.text.replace(".", " ").replace(",", " ")

            relation_pairs, from_entities, to_entities = {}, {}, {}

            for relation in sentence.relations:
                relation_from = relation.from_phrase
                word_from, from_entities = self.update_transformer_entity(
                    relation_from, tokenized_sent, sent, from_entities
                )

                relation_to = relation.to_phrase
                word_to, to_entities = self.update_transformer_entity(
                    relation_to, tokenized_sent, sent, to_entities
                )

                relation_pairs[(word_from, word_to)] = relation.label

                sent_joined = sent_joined.replace(word_from, "").replace(word_to, "")
                original_sent = original_sent.replace(relation_from.text, "").replace(
                    relation_to.text, ""
                )
                if word_from not in sentence_ent:
                    sentence_ori.append(relation_from.text)
                    sentence_ent.append(word_from)
                if word_to not in sentence_ent:
                    sentence_ori.append(relation_to.text)
                    sentence_ent.append(word_to)

            token1_pos, token2_pos = 0, 0
            sentence_ent = sent_joined.split() + sentence_ent
            sentence_ori = original_sent.split() + sentence_ori
            for from_word in sentence_ent:
                from_entity, from_vector = from_entities.get(from_word, default_pair)
                if from_word.startswith("Ġ"):
                    original_token1: str = sentence_ori[token1_pos]
                    token1_pos += 1
                if from_vector.all() == 0:
                    from_vector = sent[token1_pos + 1]

                for to_word in sentence_ent:
                    if from_word == to_word:
                        continue

                    to_entity, to_vector = from_entities.get(to_word, default_pair)
                    if to_word.startswith("Ġ"):
                        original_token2: str = sentence_ori[token2_pos]
                        token2_pos += 1
                    if to_vector.all() == 0:
                        to_vector = sent[token2_pos + 1]

                    pair = (from_word, to_word)
                    relation = relation_pairs.get(pair, self._default_tag)
                    relation = pd.Series(
                        {
                            "token1": from_word,
                            "original_token1": original_token1,
                            "vector1": from_vector,
                            "tag1": from_entity,
                            "token2": to_word,
                            "original_token2": original_token2,
                            "vector2": to_vector,
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

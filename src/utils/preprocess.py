#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import numpy as np
import warnings
from abc import abstractmethod
from ehealth.anntools import Collection
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords
import re
import unidecode

from logger.logger import Logger

warnings.simplefilter(action="ignore", category=FutureWarning)


class Preprocessor:

    _instance = None
    _n_classes = None
    _default_tag = "O"
    _models_dir = "..\\dataset"

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if Preprocessor._instance is None:
            Preprocessor()
        return Preprocessor._instance

    def __init__(self) -> None:
        import stanfordnlp

        if Preprocessor._instance is not None:
            raise Exception

        self._le = LabelEncoder()

        stanfordnlp.download("es", self._models_dir)  # Download the Spanish models
        self._tagger = stanfordnlp.Pipeline(
            processors="tokenize,pos",
            models_dir=self._models_dir,
            treebank="es_ancora",
            lang="es",
            pos_batch_size=3000,
            verbose=False,
        )

        Preprocessor._instance = self

    @abstractmethod
    def process_content(self, path: Path) -> pd.DataFrame:
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
    def preprocess(text: str, without_stopwords: bool = False) -> str:
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

        stemmer = SnowballStemmer("spanish")
        tokens_stemmed = [stemmer.stem(token) for token in tokens]

        return " ".join(tokens_stemmed)

    @staticmethod
    def prepare_labels(y_train: np.ndarray, y_test: np.ndarray) -> np.ndarray:
        from keras.utils.np_utils import to_categorical

        return to_categorical(
            y_train, num_classes=Preprocessor._n_classes
        ), to_categorical(y_test, num_classes=Preprocessor._n_classes)

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

    def data_augmentation(
        self, df: pd.DataFrame, type: str, cls: str = "B-Reference", n: int = 1
    ) -> pd.DataFrame:
        import nlpaug.augmenter.word as naw
        from utils.fileshandler import FilesHandler

        self._logger.info(
            f"Generating Data Augmentation tokens for class {cls} with technique {type}"
        )

        df_cls = df[df.tag == cls]
        if type == "back_translation":
            aug = naw.BackTranslationAug(
                from_model_name="Helsinki-NLP/opus-mt-es-en",
                to_model_name="Helsinki-NLP/opus-mt-en-es",
            )

        if type == "synonym":
            aug = naw.SynonymAug(lang="spa")

        new_df = pd.DataFrame()
        index: int = 0
        for _, row in df_cls.iterrows():
            augments = aug.augment(data=row.token, num_thread=10, n=n)
            for augment in augments:
                token = augment.replace(".", "")
                entity = pd.Series(
                    {
                        "token": token,
                        "original_token": augment,
                        "tag": cls,
                        "pos_tag": row.pos_tag,
                        "sentence": row.sentence.replace(row.token, augment),
                    },
                    name=index,
                )
                self._logger.info(f"Generated word {token} for original {row.token}")
                index += 1
                new_df = new_df.append(entity)

        self._logger.info(f"Generated {len(new_df)} tokens for class {cls}")
        FilesHandler.instance().save_train_dataset(new_df, f"_aug_{type}_{cls}")
        return df

    def get_entity_positions(self, sent: str, entity: str, offset: int) -> str:
        entity = entity.split()
        positions = []
        start = 0
        for i in range(len(entity)):
            init_pos = sent.find(entity[i], start)
            end_pos = init_pos + len(entity[i])
            positions.append(str(init_pos + offset) + ":" + str(end_pos + offset))
            start = end_pos

        return ";".join(positions)

    def trim(self, word: str) -> str:
        return re.sub(r"^\W+|\W+$", "", word)

    def trim_sentence(self, sent: str) -> str:
        trimmed_sent = []
        for word in sent.split():
            new_word = re.sub(r"^\W+|\W+$", "", word)
            trimmed_sent.append(new_word)
        return " ".join(trimmed_sent)


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
        sent_id: int = 0

        sentences = []
        for sentence in collection.sentences:

            if sentence.text in sentences or sentence.keyphrases == []:
                continue
            sentences.append(sentence.text)

            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = keyphrases.text.split()

                for i in range(len(entities)):
                    tag = "B-" if i == 0 else "I-"
                    old_entity = sentence_entities.get(entities[i], [])
                    old_entity.append(tag + keyphrases.label)
                    sentence_entities[entities[i]] = old_entity

            position = "0:0"
            for word in sentence.text.split():
                # Avoid empty words
                if word == "," or word == ".":
                    continue
                real_word = self.trim(word)

                tag = sentence_entities.get(real_word, [self._default_tag])
                offset = int(position.split(":")[-1])
                position = self.get_entity_positions(sentence.text, real_word, offset)
                word = pd.Series(
                    {
                        "token": real_word,
                        "original_token": real_word,
                        "tag": max(set(tag), key=tag.count),
                        "pos_tag": Preprocessor.instance()
                        ._tagger(real_word)
                        .sentences[0]
                        .words[0]
                        .upos,
                        "positions": position,
                        "sentence": sentence.text,
                    },
                    name=index,
                )
                index += 1
                df = df.append(word)

            self._logger.info(f"Finished sentence {sent_id} of {len(collection)}")
            sent_id += 1

        self._logger.info(f"Training completed: Stored {index} words.")
        return df


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

    def join_sublist(self, sublist: list, l: list) -> list:
        def _find_sub_list(sublist: list, l: list) -> tuple:
            sll = len(sublist)
            for ind in (i for i, e in enumerate(l) if e == sublist[0]):
                if l[ind : ind + sll] == sublist:
                    return ind, ind + sll - 1
            return 0, 0

        if len(sublist) > 1:
            init, end = _find_sub_list(sublist, l)
            l[init : end + 1] = [" ".join(l[init : end + 1])]
        return l

    def add_spaced_entities(self, sent_list: list, entity_list: list) -> list:
        added_sent = sent_list
        sent = " ".join(sent_list)
        for entity in entity_list:
            if entity not in sent:
                added_sent.append(entity)
        return added_sent

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

            relation_pairs, entities = {}, {}
            sent: str = sentence.text
            sent_tokens: list = self.trim_sentence(sentence.text).split()

            for relation in sentence.relations:
                relation_from = relation.from_phrase.text
                relation_to = relation.to_phrase.text

                sent_tokens = self.join_sublist(relation_from.split(), sent_tokens)
                sent_tokens = self.join_sublist(relation_to.split(), sent_tokens)

                relation_pairs[(relation_from, relation_to)] = relation.label
                entities[relation_from] = relation.from_phrase.label
                entities[relation_to] = relation.to_phrase.label

            from_position, to_position = ["0:0"], ["0:0"]
            last_from_word = sent_tokens[0]
            from_offset, to_offset = 0, 0
            sent_tokens = self.add_spaced_entities(sent_tokens, list(entities.keys()))

            for from_word in sent_tokens:
                from_entity = entities.get(from_word, self._default_tag)
                # If we iter the same word, dont update offset
                if last_from_word != from_word:
                    from_offset = int(from_position[-1].split(":")[-1])
                from_pos = self.get_entity_positions(sent, from_word, from_offset)

                for to_word in sent_tokens:
                    if from_word == to_word:
                        continue

                    to_entity = entities.get(to_word, self._default_tag)

                    to_offset = int(to_position[-1].split(":")[-1])
                    to_pos = self.get_entity_positions(sent, to_word, to_offset)

                    pair = (from_word, to_word)
                    relation = relation_pairs.get(pair, self._default_tag)
                    relation = pd.Series(
                        {
                            "token1": from_word,
                            "original_token1": from_word,
                            "tag1": from_entity,
                            "position1": from_pos,
                            "token2": to_word,
                            "original_token2": to_word,
                            "tag2": to_entity,
                            "position2": to_pos,
                            "tag": relation,
                            "sentence": sentence.text,
                        },
                        name=index,
                    )
                    index += 1
                    df = df.append(relation)
                last_from_word = from_word

            self._logger.info(f"Finished sentence {sent_id} of {len(collection)}")
            sent_id += 1

        self._logger.info(f"Training completed: Stored {index} word pairs.")
        return df

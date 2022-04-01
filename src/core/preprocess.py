#!/usr/bin/env python

from sklearn.preprocessing import LabelEncoder
from pathlib import Path
import pandas as pd
import numpy as np
import time
import warnings

from logger.logger import Logger

warnings.simplefilter(action="ignore", category=FutureWarning)


class Preprocessor:

    _instance = None
    _n_classes = None

    @staticmethod
    def instance():
        if Preprocessor._instance is None:
            Preprocessor()
        return Preprocessor._instance

    def __init__(self) -> None:
        if Preprocessor._instance is not None:
            raise Exception

        self._logger = Logger.instance()
        self._le = LabelEncoder()
        self._le.fit(
            [
                "B-Action",
                "B-Concept",
                "B-Predicate",
                "B-Reference",
                "I-Action",
                "I-Concept",
                "I-Predicate",
                "I-Reference",
                "O",
            ]
        )
        Preprocessor._instance = self

    @staticmethod
    def train_test_split(
        train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:
        from core.featureshandler import FeaturesHandler

        # Transform labels
        y_train, y_test = Preprocessor.instance().encode_labels(train_df, test_df)

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
        from featureshandler import FeaturesHandler
        import dask.dataframe as dd
        from dask import delayed, compute

        df.drop(y_column, axis=1, inplace=True)

        delayed_results = []
        ddf = dd.from_pandas(df, chunksize=chunk_size)
        partitions = len(ddf.to_delayed())
        part_n = 1
        for part in ddf.to_delayed():
            df = part.compute()
            Preprocessor.instance()._logger.info(
                f"Computing part of dataframe, part {part_n} of {partitions}"
            )
            part_n += 1
            X = delayed(FeaturesHandler.instance().handle_features)(df)
            delayed_results.append(X)

        start = time.time()
        results = compute(*delayed_results, scheduler="threads")
        Preprocessor.instance()._logger.info(
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
        alphanumeric_text = re.sub("[^0-9a-zA-Z]+", " ", unaccented_string)
        tokens = alphanumeric_text.lower().split()

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
    def process_content(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            for relation in sentence.relations:
                relation = pd.Series(
                    {
                        "word1": Preprocessor.preprocess(
                            relation.from_phrase.text.lower()
                        ),
                        "tag1": relation.from_phrase.label,
                        "word2": Preprocessor.preprocess(
                            relation.to_phrase.text.lower()
                        ),
                        "tag2": relation.to_phrase.label,
                        "relation": relation.label,
                        "sentence": Preprocessor.preprocess(sentence.text),
                    },
                    name=index,
                )
                index += 1
                df = df.append(relation)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} relation pairs."
        )
        return df

    @staticmethod
    def process_content_as_IOB_format(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_IOB_format")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = keyphrases.text.split()
                for i in range(len(entities)):
                    tag = "B-" if i == 0 else "I-"
                    old_entity = sentence_entities.get(entities[i], [])
                    old_entity.append(tag + keyphrases.label)
                    sentence_entities[entities[i]] = old_entity

            words = sentence.text.split()
            for word in words:
                tag = sentence_entities.get(word, ["O"])
                word = pd.Series(
                    {
                        "word": word,
                        "tag": max(set(tag), key=tag.count),
                        "sentence": sentence.text,
                    },
                    name=index,
                )
                index += 1
                df = df.append(word)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} words."
        )
        return df

    @staticmethod
    def process_content_as_BILUOV_format(path: Path) -> pd.DataFrame:
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_BILUOV_format")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        for sentence in collection.sentences:
            sentence_entities = {}
            for keyphrases in sentence.keyphrases:
                entities = Preprocessor.preprocess(keyphrases.text).split()
                for i in range(len(entities)):
                    # 1 word entity
                    if len(entities) == 1:
                        tag = "U-"
                    # More than 1 word entities
                    elif i == 0:
                        tag = "B-"
                    # Last word in entity
                    elif i == len(entities) - 1:
                        tag = "L-"
                    # Inner word
                    else:
                        tag = "I-"
                    # Overlapped word
                    if sentence_entities.get(entities[i], -1) != -1:
                        tag = "V-"
                    sentence_entities[entities[i]] = tag + keyphrases.label

            words = Preprocessor.preprocess(sentence.text).split()
            for word in words:
                tag = sentence_entities.get(word, "O")
                word = pd.Series(
                    {"word": word, "tag": tag},
                    name=index,
                )
                index += 1
                df = df.append(word)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} words."
        )
        return df

    @staticmethod
    def process_content_as_sentences(
        path: Path, transformer_type: str, as_id: bool = False
    ) -> pd.DataFrame:
        from ehealth.anntools import Collection
        from core.embeddinghandler import Embedding, TransformerEmbedding

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_sentences")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        for sentence in collection.sentences:
            prep_sent = Preprocessor.preprocess(sentence.text)
            if as_id:
                sent = TransformerEmbedding.instance().tokenize_input_ids(prep_sent)
            else:
                sent = TransformerEmbedding.instance().sentence_vector(prep_sent)
            tokenized_sent = TransformerEmbedding.instance().tokenize(prep_sent)
            sentence_entities = {}
            for keyphrase in sentence.keyphrases:
                entities = keyphrase.text.split()
                for i in range(len(entities)):
                    entity_word = TransformerEmbedding.instance().tokenize(entities[i])
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + keyphrase.label
                    for j in range(len(entity_word)):
                        if j != 0:
                            break
                        sentence_entities[entity_word[j]] = tag
            for i in range(len(tokenized_sent)):
                tag = sentence_entities.get(tokenized_sent[i], "O")
                word = pd.Series(
                    {
                        "token": tokenized_sent[i],
                        "vector": sent[i + 1],
                        "tag": tag,
                        "sentence": prep_sent,
                    },
                    name=index,
                )
                index += 1
                df = df.append(word)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} words."
        )
        return df

    def process_content_as_sentences_tensor(
        path: Path, transformer_type: str
    ) -> pd.DataFrame:
        from ehealth.anntools import Collection
        from embeddinghandler import Embedding, TransformerEmbedding

        collection = Collection().load_dir(path)
        Preprocessor.instance()._logger.info(
            f"Loaded {len(collection)} sentences for fitting."
        )
        Preprocessor.instance()._logger.info(f"process_content_as_sentences_tensor")

        df: pd.DataFrame = pd.DataFrame()
        index: int = 0

        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        for sentence in collection.sentences:
            sent = TransformerEmbedding.instance().sentence_vector(sentence.text)
            tokenized_sent = TransformerEmbedding.instance().tokenize(sentence.text)
            sentence_entities = {}
            labels_vector = []
            for keyphrase in sentence.keyphrases:
                entities = keyphrase.text.split()
                for i in range(len(entities)):
                    entity_word = TransformerEmbedding.instance().tokenize(entities[i])
                    tag = "B-" if i == 0 else "I-"
                    tag = tag + keyphrase.label
                    for j in range(len(entity_word)):
                        if j != 0:
                            break
                        sentence_entities[entity_word[j]] = tag
            for i in range(len(tokenized_sent)):
                tag = sentence_entities.get(tokenized_sent[i], "O")
                labels_vector.append(tag)
            serie = pd.Series(
                {
                    "sentence": sentence.text,
                    "vector": sent,
                    "tag": labels_vector,
                },
                name=index,
            )
            index += 1
            df = df.append(serie)

        Preprocessor.instance()._logger.info(
            f"Training completed: Stored {index} sentences."
        )
        return df

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
        from model.coremodel import CoreModel

        Preprocessor.instance()._logger.info(f"Transforming {y_column} into labels")
        y_train = self._le.fit_transform(train_df[y_column].values)
        y_test = self._le.transform(test_df[y_column].values)

        CoreModel.instance().set_labels(list(self._le.classes_))
        Preprocessor._n_classes = len(list(self._le.classes_))

        return y_train, y_test

    def encode_2D_labels(
        self, train_df: pd.DataFrame, test_df: pd.DataFrame, y_column: str = "tag"
    ) -> np.ndarray:

        from model.coremodel import CoreModel

        Preprocessor.instance()._logger.info(f"Transforming {y_column} into 2D labels")
        y_train = train_df[y_column].apply(self._le.transform)
        y_test = test_df[y_column].apply(self._le.transform)

        CoreModel.instance().set_labels(list(self._le.classes_))
        Preprocessor._n_classes = len(list(self._le.classes_))

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

        from embeddinghandler import Embedding, TransformerEmbedding

        Logger.instance().info(
            f"Starting data augmentation for the {last_n_classes}th least represented classes"
        )

        augmented_df: pd.DataFrame = df.copy()
        if not Embedding.trained():
            TransformerEmbedding.instance().build_transformer(type=transformer_type)

        tags = df.tag.value_counts().index.tolist()[-last_n_classes:]
        if classes_to_augment:
            tags = classes_to_augment

        Logger.instance().info(
            "Data augmentation for classes: " + ", ".join(classes_to_augment)
        )

        index: int = df.iloc[-1].name
        for tag in tags:
            Logger.instance().info(f"Data augmentation for class {tag}")
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

        Logger.instance().info(
            f"Finished data augmentation, added {index - df.iloc[-1].name} new synsets"
        )
        return augmented_df

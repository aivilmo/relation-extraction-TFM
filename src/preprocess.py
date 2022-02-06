#!/usr/bin/env python

from pathlib import Path
from pyexpat import features
import re
from typing_extensions import Self
import pandas as pd


class Preprocessor:
    @staticmethod
    def train_test_split(train_df: pd.DataFrame, test_df: pd.DataFrame):
        from featureshandler import FeaturesHandler
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder

        le = LabelEncoder()
        ohe = OneHotEncoder(sparse=False)

        train_df_old, test_df_old = train_df.copy(), test_df.copy()
        train_df = train_df.drop("tag", axis=1).drop("filename", axis=1)
        test_df = test_df.drop("tag", axis=1).drop("filename", axis=1)

        X_train = ohe.fit_transform(train_df.word.values.reshape(-1, 1))
        y_train = le.fit_transform(train_df_old.tag.values).reshape(-1, 1)

        X_test = ohe.transform(test_df.word.values.reshape(-1, 1))
        y_test = le.transform(test_df_old.tag.values).reshape(-1, 1)

        return X_train, X_test, y_train, y_test

    @staticmethod
    def preprocess(text: str, stopwords: bool = False) -> list:
        from nltk.stem import WordNetLemmatizer, PorterStemmer
        from nltk.corpus import stopwords
        import re
        import unidecode

        unaccented_string = unidecode.unidecode(text)
        alphanumeric_text = re.sub("[^0-9a-zA-Z]+", " ", unaccented_string)

        # Conversión texto a minúsculas y tokenización a lista de palabras
        tokens = alphanumeric_text.lower().split()

        # Eliminación las stopwords
        if stopwords:
            stop_words = set(stopwords.words("spanish"))
            tokens = [token for token in tokens if token not in stop_words]

        # Lematización
        wordnet_lemmatizer = WordNetLemmatizer()
        tokens_lemmas = [wordnet_lemmatizer.lemmatize(token) for token in tokens]

        # Stemming
        stemmer = PorterStemmer()
        tokens_stemmed = [stemmer.stem(token) for token in tokens_lemmas]

        return " ".join(tokens_stemmed)

    @staticmethod
    def process_content(path: Path):
        from ehealth.anntools import Collection

        collection = Collection().load_dir(path)
        print(f"Loaded {len(collection)} sentences for fitting.")

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

        print(f"Training completed: Stored {index} relation pairs.")
        return df

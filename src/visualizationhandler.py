#!/usr/bin/env python

from unicodedata import name
from numpy import int32
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class VisualizationHandler:

    palette = sns.color_palette("Paired", 9)
    sns.set(rc={"figure.figsize": (12, 6)})

    @staticmethod
    def visualice_relations(df: pd.DataFrame) -> None:
        sns.countplot(
            x="relation",
            data=df,
            order=df.relation.value_counts().index,
            palette=VisualizationHandler.palette,
        ).set_title("Relations")

        plt.show()

    @staticmethod
    def visualice_most_common_words(df: pd.DataFrame, n_words: int32) -> None:
        _, ax = plt.subplots(1, 2)
        VisualizationHandler.palette

        count1 = sns.countplot(
            x="word1",
            data=df,
            order=df.word1.value_counts().index[:n_words],
            ax=ax[0],
            palette=VisualizationHandler.palette,
        )
        count2 = sns.countplot(
            x="word2",
            data=df,
            order=df.word2.value_counts().index[:n_words],
            ax=ax[1],
            palette=VisualizationHandler.palette,
        )

        count1.set_title("Word1 frecuencies")
        count2.set_title("Word2 frecuencies")

        count1.set_xticklabels(count1.get_xticklabels(), rotation=45)
        count2.set_xticklabels(count2.get_xticklabels(), rotation=45)
        plt.show()

    @staticmethod
    def visualice_most_common_relations(df: pd.DataFrame) -> None:
        pass

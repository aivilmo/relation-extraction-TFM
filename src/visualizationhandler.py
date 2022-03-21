#!/usr/bin/env python

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class VisualizationHandler:

    _palette = sns.color_palette("Paired", 9)
    sns.set(rc={"figure.figsize": (12, 7)})

    @staticmethod
    def visualice_tags(df: pd.DataFrame) -> None:
        sns.countplot(
            x="tag",
            data=df,
            order=df.tag.value_counts().index,
            palette=VisualizationHandler._palette,
        ).set_title("Tags")
        print(df.tag.value_counts())
        plt.show()

    @staticmethod
    def visualice_relations(df: pd.DataFrame) -> None:
        sns.countplot(
            x="relation",
            data=df,
            order=df.relation.value_counts().index,
            palette=VisualizationHandler._palette,
        ).set_title("Relations")

        plt.show()

    @staticmethod
    def visualice_most_common_words(df: pd.DataFrame, n_words: int) -> None:
        _, ax = plt.subplots(1, 2)

        count1 = sns.countplot(
            x="word1",
            data=df,
            order=df.word1.value_counts().index[:n_words],
            ax=ax[0],
            palette=VisualizationHandler._palette,
        )
        count2 = sns.countplot(
            x="word2",
            data=df,
            order=df.word2.value_counts().index[:n_words],
            ax=ax[1],
            palette=VisualizationHandler._palette,
        )

        count1.set_title("Word1 frecuencies")
        count2.set_title("Word2 frecuencies")

        count1.set_xticklabels(count1.get_xticklabels(), rotation=45)
        count2.set_xticklabels(count2.get_xticklabels(), rotation=45)
        plt.show()

    @staticmethod
    def visualice_most_common_relations(
        df: pd.DataFrame, n_relation: int, with_relation: bool = False
    ) -> None:
        relations = df.word1 + "-"
        if with_relation:
            relations += df.relation + "-"
        df["relations"] = relations + df.word2

        sns.set(font_scale=0.90)
        sns.countplot(
            x="relations",
            data=df,
            order=df.relations.value_counts().index[:n_relation],
            palette=VisualizationHandler._palette,
        ).set_title("Words related")

        plt.xticks(rotation=25)
        plt.show()

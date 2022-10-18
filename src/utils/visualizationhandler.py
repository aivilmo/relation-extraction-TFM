#!/usr/bin/env python

import pandas as pd
from regex import P
import seaborn as sns
import matplotlib.pyplot as plt


class VisualizationHandler:

    _palette = sns.color_palette("Paired", 9)
    sns.set(rc={"figure.figsize": (12, 7)})

    @staticmethod
    def visualice_tags(df: pd.DataFrame) -> None:
        df["tag"] = df["tag"].apply(lambda row: row[2:] if "-" in row else row)

        tags = sns.countplot(
            x="tag",
            data=df,
            order=df.tag.value_counts().index,
            palette=VisualizationHandler._palette,
        )
        print(df.tag.value_counts())
        tags.set_title("Entity types")
        tags.set_xticklabels(tags.get_xticklabels(), rotation=25, fontsize=20)
        plt.show()

    @staticmethod
    def visualice_relations(df: pd.DataFrame) -> None:
        df = df[df.tag != "O"]

        relations = sns.countplot(
            x="tag",
            data=df,
            order=df.tag.value_counts().index,
            palette=VisualizationHandler._palette,
        )
        print(df.tag.value_counts())
        relations.set_title("Relation types")
        relations.set_xticklabels(relations.get_xticklabels(), rotation=25, fontsize=20)
        plt.show()

    @staticmethod
    def visualice_most_common_words(df: pd.DataFrame, n_words: int) -> None:
        _, ax = plt.subplots(1, 2)

        count1 = sns.countplot(
            x="token1",
            data=df,
            order=df.token1.value_counts().index[:n_words],
            ax=ax[0],
            palette=VisualizationHandler._palette,
        )
        count2 = sns.countplot(
            x="token2",
            data=df,
            order=df.token2.value_counts().index[:n_words],
            ax=ax[1],
            palette=VisualizationHandler._palette,
        )

        count1.set_title("token1 frecuencies")
        count2.set_title("token2 frecuencies")

        count1.set_xticklabels(count1.get_xticklabels(), rotation=45)
        count2.set_xticklabels(count2.get_xticklabels(), rotation=45)
        plt.show()

    @staticmethod
    def visualice_most_common_word(df: pd.DataFrame, n_words: int) -> None:
        from nltk.corpus import stopwords

        stop_words = set(stopwords.words("spanish") + stopwords.words("english"))
        df["reduced_token"] = df["token"].apply(
            lambda row: row.lower() if row.lower() not in stop_words else "-1"
        )
        df = df[df.reduced_token != "-1"]

        print(df.reduced_token.value_counts()[0:15])
        count = sns.countplot(
            x="reduced_token",
            data=df,
            order=df.reduced_token.value_counts().index[:n_words],
            palette=VisualizationHandler._palette,
        )

        count.set_title("Tokens Frecuencies")
        count.set_xticklabels(count.get_xticklabels(), rotation=25, fontsize=17)
        plt.show()

    @staticmethod
    def visualice_most_common_relations(
        df: pd.DataFrame, n_relation: int, with_relation: bool = False
    ) -> None:
        relations = df.token1 + "-"
        if with_relation:
            relations += df.tag + "-"
        df["relations"] = relations + df.token2

        sns.set(font_scale=0.90)
        sns.countplot(
            x="relations",
            data=df,
            order=df.relations.value_counts().index[:n_relation],
            palette=VisualizationHandler._palette,
        ).set_title("Words related")

        plt.xticks(rotation=25)
        plt.show()

    @staticmethod
    def visualice_relations_tags(df: pd.DataFrame) -> None:
        df = df[df.tag != "O"]
        df = df.drop(
            columns=[
                "token1",
                "original_token1",
                "position1",
                "token2",
                "original_token2",
                "position2",
                "sentence",
                "predicted_tag",
            ],
            errors="ignore",
        )
        print(df.value_counts())
        print()
        df = df.replace(to_replace="Action", value="A")
        df = df.replace(to_replace="Predicate", value="P")
        df = df.replace(to_replace="Reference", value="R")
        df = df.replace(to_replace="Concept", value="C")

        df.groupby("tag1").sum().reset_index().melt(id_vars="tag1")
        df["entities"] = df["tag1"].astype(str) + "-" + df["tag2"]
        df = df.drop(columns=["tag1", "tag2"])

        df = df.value_counts().reset_index(name="counts")
        counts = df.groupby("tag").sum().reset_index()

        index = 0
        df_copy = pd.DataFrame()
        for _, row in df.iterrows():
            series = pd.Series(
                {
                    "percent": (
                        row.counts / counts[counts.tag == row.tag].counts * 100
                    ).values[0]
                },
                name=index,
            )
            index += 1
            df_copy = df_copy.append(series)
        df["percent"] = df_copy.percent
        # flights_wide = df.pivot("entities", "tag", "counts")
        flights_wide = df.pivot("entities", "tag", "percent")
        sns.lineplot(data=flights_wide)
        plt.show()

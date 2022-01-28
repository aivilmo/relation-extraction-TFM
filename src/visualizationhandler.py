#!/usr/bin/env python

from numpy import int32
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class VisualizationHandler:
    @staticmethod
    def visualice_relations(df: pd.DataFrame) -> None:
        sns.countplot(x="relation", data=df)
        plt.show()

    @staticmethod
    def visualice_most_common_words(df: pd.DataFrame, n_words: int32) -> None:
        df: pd.DataFrame = (
            df.word.value_counts()[:n_words]
            .sort_values(ascending=False)
            .to_frame()
            .reset_index()
        )
        df = df.rename(columns={"index": "word", "word": "count"})
        sns.catplot(x="word", y="count", data=df, kind="bar")
        plt.show()

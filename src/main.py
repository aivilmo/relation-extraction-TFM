#!/usr/bin/env python
from fileshandler import FilesHandler
import pandas as pd
from pathlib import Path
from typing_extensions import Self
import argparse


class Main:

    _instance = None

    def __new__(cls: type[Self], *args, **kwargs) -> Self:
        if not isinstance(cls._instance, cls):
            cls._instance = object.__new__(cls, *args, **kwargs)
        return cls._instance

    def __init__(self) -> None:
        self.__path: str = (
            "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
        )
        self.__output_train: str = "data\\train.pkl"
        self.__output_test: str = "data\\test.pkl"
        self.__dataset_train: pd.DataFrame = None
        self.__dataset_test: pd.DataFrame = None

    def main(self) -> None:
        from argsparser import ArgsParser

        args = ArgsParser.get_args()
        self.__get_datasets(args)

        if args.visualization:
            self.__handleVisualizations()
        elif args.train:
            self.__handleTrain()

    def __handleVisualizations(self) -> None:
        from visualizationhandler import VisualizationHandler

        VisualizationHandler.visualice_relations(self.__dataset_train)
        VisualizationHandler.visualice_most_common_words(
            self.__dataset_train, n_words=10
        )
        VisualizationHandler.visualice_most_common_relations(
            self.__dataset_train, n_relation=10, with_relation=True
        )

    def __handleTrain(self) -> None:
        from coremodel import CoreModel
        from preprocess import Preprocessor
        from sklearn.linear_model import Perceptron

        prep = Preprocessor()
        X_train, X_test, y_train, y_test = prep.train_test_split(
            self.__dataset_train, self.__dataset_test
        )

        core = CoreModel()
        core.set_model(Perceptron())
        core.fit_model(X_train, y_train)
        core.train_model()
        core.test_model(X_test, y_test)

    def __get_datasets(self, args: argparse.Namespace) -> None:
        if args.load:
            self.__dataset_train = FilesHandler.load_dataset(self.__output_train)
            self.__dataset_test = FilesHandler.load_dataset(self.__output_test)

        if args.generate:
            self.__dataset_train = FilesHandler.generate_dataset(
                Path(self.__path + "\\training\\"), self.__output_train
            )
            self.__dataset_test = FilesHandler.generate_dataset(
                Path(self.__path + "\\testing\\"), self.__output_test
            )


if __name__ == "__main__":
    Main().main()

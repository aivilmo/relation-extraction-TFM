#!/usr/bin/env python
from fileshandler import FilesHandler
import pandas as pd
from pathlib import Path
import argparse


class Main:

    _instance = None

    @staticmethod
    def instance():
        if Main._instance == None:
            Main()
        return Main._instance

    def __init__(self) -> None:
        if Main._instance != None:
            raise Exception

        self._path: str = (
            "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
        )
        self._output_train: str = "data\\train.pkl"
        self._output_test: str = "data\\test.pkl"
        self._dataset_train: pd.DataFrame = None
        self._dataset_test: pd.DataFrame = None
        Main._instance = self

    def main(self) -> None:
        from argsparser import ArgsParser

        args = ArgsParser.get_args()
        self._get_datasets(args)

        if args.visualization:
            self._handleVisualizations()
        elif args.train:
            self._handleTrain()

    def _handleVisualizations(self) -> None:
        from visualizationhandler import VisualizationHandler

        VisualizationHandler.visualice_relations(self._dataset_train)
        VisualizationHandler.visualice_most_common_words(
            self._dataset_train, n_words=10
        )
        VisualizationHandler.visualice_most_common_relations(
            self._dataset_train, n_relation=10, with_relation=True
        )

    def _handleTrain(self) -> None:
        from coremodel import CoreModel
        from preprocess import Preprocessor
        from sklearn.linear_model import Perceptron

        prep = Preprocessor()
        X_train, X_test, y_train, y_test = prep.train_test_split(
            self._dataset_train, self._dataset_test
        )

        core = CoreModel.instance()
        core.set_model(Perceptron())
        core.fit_model(X_train, y_train)
        core.train_model()
        core.test_model(X_test, y_test)

    def _get_datasets(self, args: argparse.Namespace) -> None:
        if args.load:
            self._dataset_train = FilesHandler.load_dataset(self._output_train)
            self._dataset_test = FilesHandler.load_dataset(self._output_test)

        if args.generate:
            self._dataset_train = FilesHandler.generate_dataset(
                Path(self._path + "\\training\\"), self._output_train
            )
            self._dataset_test = FilesHandler.generate_dataset(
                Path(self._path + "\\testing\\"), self._output_test
            )


if __name__ == "__main__":
    Main.instance().main()

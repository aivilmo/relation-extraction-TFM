#!/usr/bin/env python
import pandas as pd
from pathlib import Path
import argparse
from core.featureshandler import FeaturesHandler
from logger.logger import Logger


class Main:

    _instance = None

    @staticmethod
    def instance():
        if Main._instance is None:
            Main()
        return Main._instance

    def __init__(self) -> None:
        if Main._instance is not None:
            raise Exception

        self._path_ref: str = (
            "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
        )
        self._path_eval: str = (
            "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\eval"
        )
        self._output_train: str = "data\\ref_train.pkl"
        self._output_test: str = "data\\eval_train.pkl"
        self._dataset_train: pd.DataFrame = None
        self._dataset_test: pd.DataFrame = None
        self._args: argparse.Namespace = None
        self._logger = Logger.instance()
        Main._instance = self

    def main(self) -> None:
        from utils.argsparser import ArgsParser

        self._args = ArgsParser.get_args()
        self._get_datasets()

        if self._args.features is not None:
            FeaturesHandler.instance().features = self._args.features

        if self._args.visualization:
            self._handle_visualizations()
        if self._args.train:
            self._handle_train()
        if self._args.prepare_data:
            self._handle_prepare_data()

    def _handle_visualizations(self) -> None:
        from utils.visualizationhandler import VisualizationHandler

        VisualizationHandler.visualice_tags(self._dataset_train)

    def _handle_train(self) -> None:
        from model.coremodel import CoreModel
        from core.preprocess import Preprocessor

        X_train, X_test, y_train, y_test = Preprocessor.instance().train_test_split(
            self._dataset_train, self._dataset_test
        )

        # Train Core Model
        CoreModel.instance().start_train(X_train, X_test, y_train, y_test)

    def _handle_prepare_data(self) -> None:
        from core.preprocess import Preprocessor
        import numpy as np

        X_train, X_test, y_train, y_test = Preprocessor.instance().train_test_split(
            self._dataset_train, self._dataset_test
        )

        # Prepare to DeepModel
        y_train, y_test = Preprocessor.instance().prepare_labels(
            y_train=y_train, y_test=y_test
        )

        feat: str = "_".join(FeaturesHandler.instance().features)
        feat = feat.replace("/", "_")
        np.save("data\\X_ref_train_" + feat + ".npy", X_train)
        np.save("data\\X_eval_train_" + feat + ".npy", X_test)
        np.save("data\\y_ref_train_" + feat + ".npy", y_train)
        np.save("data\\y_eval_train_" + feat + ".npy", y_test)

        self._logger.info("Data is successfully saved in dir \\data\\")

    def _get_datasets(self) -> None:
        from utils.fileshandler import FilesHandler

        if self._args.load:
            self._dataset_train = FilesHandler.load_dataset(self._output_train)
            self._dataset_test = FilesHandler.load_dataset(self._output_test, test=True)

        if self._args.generate:
            as_sentences: bool = False
            as_IOB: bool = True
            transformer_type: str = ""

            if self._args.features is not None and (
                "bert" in self._args.features[0] or "gpt" in self._args.features[0]
            ):
                as_sentences = True
                transformer_type = self._args.features[0]

            self._dataset_train = FilesHandler.generate_dataset(
                Path(self._path_ref + "\\training\\"),
                self._output_train,
                as_IOB=as_IOB,
                as_sentences=as_sentences,
                transformer_type=transformer_type,
            )
            self._dataset_test = FilesHandler.generate_dataset(
                Path(self._path_eval + "\\training\\scenario2-taskA\\"),
                self._output_test,
                as_IOB=as_IOB,
                as_sentences=as_sentences,
                transformer_type=transformer_type,
            )


if __name__ == "__main__":
    Main.instance().main()

#!/usr/bin/env python
import pandas as pd
import argparse
import numpy as np

from core.featureshandler import FeaturesHandler
from logger.logger import Logger


class Main:

    _instance = None
    _logger = Logger.instance()

    @staticmethod
    def instance():
        if Main._instance is None:
            Main()
        return Main._instance

    def __init__(self) -> None:
        from utils.argsparser import ArgsParser

        if Main._instance is not None:
            raise Exception

        self._dataset_train: pd.DataFrame = None
        self._dataset_test: pd.DataFrame = None
        self._args: argparse.Namespace = ArgsParser.get_args()

        Main._instance = self

    def main(self) -> None:
        FeaturesHandler.instance().features = self._args.features

        X_train, X_test, y_train, y_test = self._get_datasets()
        if self._args.visualization:
            self._handle_visualizations()
        if self._args.train:
            self._handle_train(X_train, X_test, y_train, y_test)

    def _handle_visualizations(self) -> None:
        from utils.visualizationhandler import VisualizationHandler

        VisualizationHandler.visualice_tags(self._dataset_train)

    def _handle_train(self, X_train, X_test, y_train, y_test) -> None:
        from model.coremodel import CoreModel
        from model.deepmodel import DeepModel

        model_instance = None
        model = None

        # Train Core Model
        if self._args.ml_model is not None:
            model_instance = CoreModel.instance()
            model = self._args.ml_model

        # Train Deep Model
        if self._args.dl_model is not None:
            model_instance = DeepModel.instance()
            model = self._args.dl_model

        model_instance.start_train(X_train, X_test, y_train, y_test, model)

    def _get_datasets(self) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:
        from utils.fileshandler import FilesHandler
        from core.preprocess import Preprocessor

        features: str = "_".join(self._args.features)
        features = features.replace("/", "_")
        transformer_type = features if "bert" in features else ""

        self._dataset_train, self._dataset_test = FilesHandler.load_datasets(
            transformer_type=transformer_type
        )

        if self._args.load:
            return FilesHandler.load_training_data(features)

        if self._dataset_train is None:
            Main._logger.warning(f"Datasets not found, generating for {features}")
            self._dataset_train, self._dataset_test = FilesHandler.generate_datasets(
                transformer_type=transformer_type
            )

        X_train, X_test, y_train, y_test = Preprocessor.instance().train_test_split(
            self._dataset_train, self._dataset_test
        )
        FilesHandler.save_training_data(X_train, X_test, y_train, y_test, features)
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    Main.instance().main()

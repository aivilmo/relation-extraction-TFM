#!/usr/bin/env python
import pandas as pd
import argparse
import numpy as np

from core.featureshandler import FeaturesHandler
from logger.logger import Logger
from utils.fileshandler import FilesHandler


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

        FilesHandler(self._args.task)
        FeaturesHandler(self._args.task)

        Main._instance = self

    def main(self) -> None:
        FeaturesHandler.instance().features = self._args.features

        X_train, X_test, y_train, y_test = self._get_datasets()

        if self._args.visualization:
            self._handle_visualizations()
        if self._args.train:
            self._handle_train(X_train, X_test, y_train, y_test)
        if self._args.export:
            self._handle_export()

    def _handle_visualizations(self) -> None:
        from utils.visualizationhandler import VisualizationHandler

        VisualizationHandler.visualice_tags(self._dataset_train)

    def _handle_train(self, X_train, X_test, y_train, y_test) -> None:
        from model.coremodel import CoreModel
        from model.deepmodel import DeepModel
        from model.coderencodermodel import CoderEncoderModel
        from model.transformermodel import TransformerModel

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

        # Train Sequence to Sequence Model
        if self._args.s2s_model:
            model_instance = CoderEncoderModel(self._dataset_train, self._dataset_test)

        # Train Transformer Model
        if self._args.transformer_model:
            model_instance = TransformerModel(self._dataset_train, self._dataset_test)

        if model_instance is None:
            Main._logger.warning(
                "Need select a model, with args '--ml_model', --dl_model', '--s2s_model' or 'transformer_model'"
            )
            return

        model_instance.start_training(X_train, X_test, y_train, y_test, model)

    def _handle_export(self) -> None:
        from utils.postprocess import PostProcessor

        _, self._dataset_test = FilesHandler.instance().load_datasets()
        PostProcessor(self._args.run, self._args.task)
        PostProcessor.instance().export_data_to_file(self._dataset_test)

    def get_features_names(self) -> str:
        features: str = "_".join(self._args.features)
        features = features.replace("/", "_")
        return features

    def features(self) -> str:
        return self._args.features

    def get_y_column(self) -> str:
        if self._args.s2s_model:
            return "sentence"
        return "tag"

    def _get_datasets(self) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:
        from utils.preprocess import Preprocessor

        features = self.get_features_names()
        FeaturesHandler.instance().check_features_for_task()

        (
            self._dataset_train,
            self._dataset_test,
        ) = FilesHandler.instance().load_datasets()

        if self._args.load:
            return FilesHandler.instance().load_training_data(features)

        if self._dataset_train is None or self._dataset_test is None:
            Main._logger.warning(f"Datasets not found, generating for {features}")
            (
                self._dataset_train,
                self._dataset_test,
            ) = FilesHandler.instance().generate_datasets()

        X_train, X_test, y_train, y_test = Preprocessor.instance().train_test_split(
            self._dataset_train,
            self._dataset_test,
            y_column=self.get_y_column(),
        )
        FilesHandler.instance().save_training_data(
            X_train, X_test, y_train, y_test, features
        )
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    Main.instance().main()

#!/usr/bin/env python
import pandas as pd
import argparse
import numpy as np

from logger.logger import Logger
from utils.fileshandler import FilesHandler
from utils.preprocess import Preprocessor


class Main:

    _instance = None

    _logger = Logger.instance()
    _fh_instance = FilesHandler.instance()
    _pr_instance = Preprocessor.instance()

    @staticmethod
    def instance():
        if Main._instance is None:
            Main()
        return Main._instance

    def __init__(self) -> None:
        from utils.appconstants import AppConstants

        if Main._instance is not None:
            raise Exception

        self._dataset_train: pd.DataFrame = None
        self._dataset_test: pd.DataFrame = None
        self._args: argparse.Namespace = AppConstants.instance()._args

        Main._instance = self

    def main(self) -> None:
        X_train, X_test, y_train, y_test = self._get_datasets()

        if self._args.visualization:
            self._handle_visualizations()
        if self._args.train:
            self._handle_train(X_train, X_test, y_train, y_test)
        if self._args.export:
            self._handle_export()

    def _handle_visualizations(self) -> None:
        from utils.visualizationhandler import VisualizationHandler

        # NER
        if "taskA" in self._args.task:
            VisualizationHandler.visualice_tags(self._dataset_test)
            VisualizationHandler.visualice_most_common_word(self._dataset_test, 10)
            return

        # RE
        VisualizationHandler.visualice_relations_tags(self._dataset_test)

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
            self._logger.warning(
                "Need select a model, with args '--ml_model', '--dl_model', '--s2s_model' or 'transformer_model'"
            )
            return

        model_instance.start_training(X_train, X_test, y_train, y_test, model)

    def _handle_export(self) -> None:
        from utils.postprocess import PostProcessor

        _, self._dataset_test = self._fh_instance.load_datasets()
        PostProcessor.instance().export_data_to_file(self._dataset_test)

    def _handle_data_augmentation(self) -> None:
        if self._args.data_aug != None:
            type = self._args.data_aug

            ref_i = self._fh_instance.load_augmented_dataset("I-Reference", type)
            ref_b = self._fh_instance.load_augmented_dataset("B-Reference", type)
            pred_b = self._fh_instance.load_augmented_dataset("B-Predicate", type)

            if ref_i is None:
                ref_i = self._pr_instance.data_augmentation(
                    self._dataset_train, type, "I-Reference", n=10
                )
            if ref_b is None:
                ref_b = self._pr_instance.data_augmentation(
                    self._dataset_train, type, "B-Reference"
                )
            if pred_b is None:
                pred_b = self._pr_instance.data_augmentation(
                    self._dataset_train, type, "B-Predicate"
                )

            self._dataset_train = self._dataset_train.append(ref_i, ignore_index=True)
            self._dataset_train = self._dataset_train.append(ref_b, ignore_index=True)
            self._dataset_train = self._dataset_train.append(pred_b, ignore_index=True)
            return True
        return False

    def get_features_names(self) -> str:
        features: str = "_".join(self._args.features)
        features = features.replace("/", "_").replace("\\", "_")
        return features

    def features(self) -> str:
        return self._args.features

    def get_y_column(self) -> str:
        if self._args.s2s_model:
            return "sentence"
        return "tag"

    def _get_datasets(self) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:
        from core.featureshandler import FeaturesHandler

        features = self.get_features_names()
        FeaturesHandler.instance().check_features_for_task()

        self._dataset_train, self._dataset_test = self._fh_instance.load_datasets()
        if self._handle_data_augmentation():
            features = features + f"_{self._args.data_aug}_ref_pred"

        if self._args.load:
            return self._fh_instance.load_training_data(features)

        if self._dataset_train is None or self._dataset_test is None:
            self._logger.warning(f"Datasets not found, generating for {features}")
            (
                self._dataset_train,
                self._dataset_test,
            ) = self._fh_instance.generate_datasets()

        X_train, X_test, y_train, y_test = self._pr_instance.train_test_split(
            self._dataset_train,
            self._dataset_test,
            y_column=self.get_y_column(),
        )
        self._fh_instance.save_training_data(X_train, X_test, y_train, y_test, features)
        return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    Main.instance().main()

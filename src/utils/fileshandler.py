#!/usr/bin/env python

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from logger.logger import Logger
from utils.preprocess import REPreprocessor, NERPreprocessor


class FilesHandler:

    _instance = None

    _path_ref: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
    _path_eval: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\eval"

    _IOB_output = "_IOB.pkl"

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if FilesHandler._instance is None:
            FilesHandler()
        return FilesHandler._instance

    def __init__(self) -> None:
        from utils.appconstants import AppConstants

        if FilesHandler._instance is not None:
            return

        self._task: str = AppConstants.instance()._task
        self._test_dataset: str = AppConstants.instance()._test_dataset
        self._path_output: str = "data\\" + self._task
        self._output = "train"
        if "dev" in self._test_dataset:
            self._output = "dev"
        elif "test" in self._test_dataset:
            self._output = "test"

        self._output_train: str = self._path_output + "\\ref_train_finetuned"
        self._output_test: str = self._path_output + "\\eval_" + self._output
        self._output_X_train: str = self._path_output + "\\X_ref_train_finetuned"
        self._output_X_test: str = self._path_output + "\\X_eval_" + self._output
        self._output_y_train: str = self._path_output + "\\y_ref_train"
        self._output_y_test: str = self._path_output + "\\y_eval_" + self._output

        self._in_out_main: str = "data\\scenario1-main\\ref_train_finetuned"

        FilesHandler._instance = self

    def get_dataframe(self, instance, path: Path) -> pd.DataFrame:
        return instance.process_content(path)

    def get_filename(self, filename: str) -> str:
        return filename + self._IOB_output

    def features_filename(self, features: str) -> str:
        return "_" + features + ".npy"

    def generate_datasets(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        def generate_dataset(
            path: list[Path],
            output_file: str,
        ) -> pd.DataFrame:
            self._logger.info(f"Generating DataFrame... for path {path}")

            prep_instance = None
            if "taskA" in self._task:
                prep_instance = NERPreprocessor.instance()
            if "taskB" in self._task:
                prep_instance = REPreprocessor.instance()
            if "main" in self._task:
                prep_instance = REPreprocessor.instance(True)

            df: pd.DataFrame = pd.DataFrame()
            for paths in path:
                df = df.append(self.get_dataframe(prep_instance, paths))
                print(df)
            output_file = self.get_filename(output_file)

            if not self.try_to_save_dataframe(df, output_file):
                self.try_to_create_directory()
                if not self.try_to_save_dataframe(df, output_file):
                    sys.exit()

            return df

        return generate_dataset(
            path=[
                Path(self._path_ref + "\\training\\"),
                Path(self._path_ref + "\\develop\\"),
            ],
            output_file=self._output_train,
        ), generate_dataset(
            path=[
                Path(
                    self._path_eval
                    + "\\"
                    + self._test_dataset
                    + "\\"
                    + self._task
                    + "\\"
                )
            ],
            output_file=self._output_test,
        )

    def load_datasets(self, is_main=False) -> tuple[pd.DataFrame, pd.DataFrame]:
        def load_dataset(filename: str) -> pd.DataFrame:
            filename = self.get_filename(filename)
            self._logger.info(f"Loading DataFrame from: {filename}")

            try:
                df: pd.DataFrame = pd.read_pickle(filename)

            except (FileNotFoundError, OSError) as e:
                self._logger.error(e)
                return None

            self._logger.info("DataFrame succesfully loaded")
            return df

        if not is_main:
            return load_dataset(self._output_train), load_dataset(self._output_test)
        return load_dataset(self._output_train), load_dataset(self._in_out_main)

    def save_datasets(
        self, train_dataset: pd.DataFrame, test_dataset: pd.DataFrame, is_main=False
    ) -> None:
        end_filename: str = self._IOB_output

        if is_main:
            train_dataset.to_pickle(self._in_out_main + end_filename)
        else:
            train_dataset.to_pickle(self._output_train + end_filename)
            test_dataset.to_pickle(self._output_test + end_filename)
        self._logger.info("Datasets successfully saved")

    def load_training_data(
        self, features: str, is_main=False
    ) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:

        try:
            self._logger.info(f"Loading training data...")
            filename = self.features_filename(features)

            if is_main:
                return np.load(
                    "data\\scenario1-main\\X_ref_train_finetuned_PlanTL-GOB-ES_roberta-base-biomedical-es.npy",
                    allow_pickle=True,
                )

            X_train = np.load(
                self._output_X_train + filename,
                allow_pickle=True,
            )
            X_test = np.load(
                self._output_X_test + filename,
                allow_pickle=True,
            )
            y_train = np.load(
                self._output_y_train + filename,
                allow_pickle=True,
            )
            y_test = np.load(
                self._output_y_test + filename,
                allow_pickle=True,
            )

            self._logger.info("Training data succesfully loaded")
            return X_train, X_test, y_train, y_test

        except (FileNotFoundError, OSError) as e:
            self.catch_error_loading_data(features, e)

    def save_training_data(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        features: np.ndarray,
    ) -> None:
        try:
            filename = self.features_filename(features)

            np.save(self._output_X_train + filename, X_train)
            np.save(self._output_X_test + filename, X_test)
            np.save(self._output_y_train + filename, y_train)
            np.save(self._output_y_test + filename, y_test)

            self._logger.info(f"Data is successfully saved in dir \\data\\{self._task}")

        except OSError as e:
            self._logger.error(e)
            sys.exit()

    def try_to_save_dataframe(self, df: pd.DataFrame, output_file: str) -> bool:
        try:
            df.to_pickle(output_file)
            df.to_csv(output_file.replace(".pkl", ".csv"), sep="\t")
        except (OSError, KeyError) as e:
            self._logger.error(e)
            return False

        self._logger.info(
            f"DataFrame successfully generated and saved at: {output_file}"
        )
        print(df)
        return True

    def try_to_create_directory(self) -> None:
        import os

        try:
            self._logger.warning(f"Creating directory {self._path_output}...")
            os.makedirs(self._path_output, exist_ok=True, mode=0o777)
            self._logger.info("Directory created successfully")
        except Exception as e:
            self._logger.error(e)
            self._logger.error("Creation of directory has failed")
            sys.exit()

    def catch_error_loading_data(self, features: str, e: Exception) -> None:
        self._logger.error(e)
        self._logger.error(f"Need to generate training files for features: {features}")
        self._logger.error(
            f"Run: 'python .\main.py --generate --features {features} --task {self._task}'"
        )
        sys.exit()

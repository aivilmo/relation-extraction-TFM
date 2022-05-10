#!/usr/bin/env python

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from logger.logger import Logger


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

    def __init__(self, task: str) -> None:
        if FilesHandler._instance is not None:
            return

        self._task: str = task
        self._path_output: str = "data\\" + task

        self._output_train: str = self._path_output + "\\ref_train"
        self._output_test: str = self._path_output + "\\eval_train"
        self._output_X_train: str = self._path_output + "\\X_ref_train"
        self._output_X_test: str = self._path_output + "\\X_dev_train"
        self._output_y_train: str = self._path_output + "\\y_ref_train"
        self._output_y_test: str = self._path_output + "\\y_dev_train"

        FilesHandler._instance = self

    def transformer_filename(self, transformer: str) -> str:
        return "_" + transformer.replace("/", "_") + ".pkl"

    def features_filename(self, features: str) -> str:
        return "_" + features + ".npy"

    def generate_datasets(
        self,
        transformer_type: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        def generate_dataset(
            path: Path,
            output_file: str,
            transformer_type: str = "",
        ) -> pd.DataFrame:
            from utils.preprocess import REPreprocessor, NERPreprocessor

            self._logger.info("Generating DataFrame...")

            prep_instance = None
            if "taskA" in self._task:
                prep_instance = NERPreprocessor.instance()
            if "taskB" in self._task:
                prep_instance = REPreprocessor.instance()

            if transformer_type == "":
                df: pd.DataFrame = prep_instance.process_content(path)
                output_file += self._IOB_output
            else:
                df: pd.DataFrame = prep_instance.process_content_cased_transformer(
                    path, transformer_type
                )
                output_file += self.transformer_filename(transformer_type)

            if not self.try_to_save_dataframe(df, output_file):
                self.try_to_create_directory()
                if not self.try_to_save_dataframe(df, output_file):
                    sys.exit()

            return df

        return generate_dataset(
            path=Path(self._path_ref + "\\training\\"),
            output_file=self._output_train,
            transformer_type=transformer_type,
        ), generate_dataset(
            path=Path(self._path_eval + "\\training\\" + self._task + "\\"),
            output_file=self._output_test,
            transformer_type=transformer_type,
        )

    def load_datasets(self, transformer_type="") -> tuple[pd.DataFrame, pd.DataFrame]:
        def load_dataset(filename: str, transformer_type="") -> pd.DataFrame:
            if transformer_type == "":
                filename = filename + self._IOB_output
            else:
                filename = filename + self.transformer_filename(transformer_type)

            try:
                self._logger.info(f"Loading DataFrame from: {filename}")
                df: pd.DataFrame = pd.read_pickle(filename)

            except (FileNotFoundError, OSError) as e:
                self._logger.error(e)
                return None

            self._logger.info("DataFrame succesfully loaded")
            return df

        return load_dataset(self._output_train, transformer_type), load_dataset(
            self._output_test, transformer_type
        )

    def save_datasets(
        self,
        train_dataset: pd.DataFrame,
        test_dataset: pd.DataFrame,
        transformer_type: str = "",
    ) -> None:
        transformer_filename = self.transformer_filename(transformer_type)
        if transformer_type == "":
            train_dataset.to_pickle(self._output_train + self._IOB_output)
            test_dataset.to_pickle(self._output_test + self._IOB_output)
        else:
            train_dataset.to_pickle(self._output_train + transformer_filename)
            test_dataset.to_pickle(self._output_test + transformer_filename)
        self._logger.info("Datasets successfully saved")

    def load_training_data(
        self,
        features: str,
    ) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:

        try:
            self._logger.info(f"Loading training data...")
            filename = self.features_filename(features)

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

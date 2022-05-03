#!/usr/bin/env python

from pathlib import Path
import sys
import pandas as pd
import numpy as np

from logger.logger import Logger
from utils.preprocess import Preprocessor


class FilesHandler:

    _instance = None

    _path_ref: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
    _path_eval: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\eval"

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if FilesHandler._instance is None:
            FilesHandler()
        return FilesHandler._instance

    def __init__(self, task: str) -> None:
        if FilesHandler._instance is not None:
            raise Exception

        self._task: str = task
        self._path_output: str = "data\\" + task

        self._output_train: str = self._path_output + "\\ref_train"
        self._output_test: str = self._path_output + "\\eval_train"
        self._output_X_train: str = self._path_output + "\\X_ref_train"
        self._output_X_test: str = self._path_output + "\\X_dev_train"
        self._output_y_train: str = self._path_output + "\\y_ref_train"
        self._output_y_test: str = self._path_output + "\\y_dev_train"

        FilesHandler._instance = self

    @staticmethod
    def try_to_save_dataframe(df: pd.DataFrame, output_file: str) -> bool:
        try:
            df.to_pickle(output_file)
            df.to_csv(output_file.replace(".pkl", ".csv"), sep="\t")
        except (OSError, KeyError) as e:
            FilesHandler._logger.error(e)
            return False

        FilesHandler._logger.info(
            f"DataFrame successfully generated and saved at: {output_file}"
        )
        print(df)
        return True

    def try_to_create_directory(self) -> None:
        import os

        try:
            FilesHandler._logger.warning(f"Creating directory {self._path_output}...")
            os.makedirs(self._path_output, exist_ok=True, mode=0o777)
            FilesHandler._logger.info("Directory created successfully")
        except Exception as e:
            FilesHandler._logger.error(e)
            FilesHandler._logger.error("Creation of directory has failed")
            sys.exit()

    def catch_error_loading_data(self, features: str, e: Exception) -> None:
        FilesHandler._logger.error(e)
        FilesHandler._logger.error(
            f"Need to generate training files for features: {features}"
        )
        FilesHandler._logger.error(
            f"Run: 'python .\main.py --generate --features {features} --task {self._task}'"
        )
        sys.exit()

    def generate_datasets(
        self,
        as_IOB: bool = True,
        as_BILUOV: bool = False,
        transformer_type: str = "",
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        def generate_dataset(
            path: Path,
            output_file: str,
            as_IOB: bool = True,
            as_BILUOV: bool = False,
            transformer_type: str = "",
        ) -> pd.DataFrame:

            FilesHandler._logger.info("Generating DataFrame...")

            if transformer_type != "":
                if "taskB" in self._task:
                    df: pd.DataFrame = (
                        Preprocessor.process_content_as_IOB_with_relations(
                            path, transformer_type=transformer_type
                        )
                    )
                else:
                    df: pd.DataFrame = Preprocessor.process_content_as_sentences(
                        path, transformer_type=transformer_type
                    )
                transformer_type = transformer_type.replace("/", "_")
                output_file += "_" + transformer_type + ".pkl"

            elif as_IOB:
                df: pd.DataFrame = Preprocessor.process_content_as_IOB_format(path)
                output_file += "_IOB.pkl"

            # DEPRECATED
            elif as_BILUOV:
                df: pd.DataFrame = Preprocessor.process_content_as_BILUOV_format(path)
                output_file += "_BILUOV.pkl"

            if not FilesHandler.try_to_save_dataframe(df, output_file):
                self.try_to_create_directory()
                if not FilesHandler.try_to_save_dataframe(df, output_file):
                    sys.exit()

            return df

        return generate_dataset(
            path=Path(self._path_ref + "\\training\\"),
            output_file=self._output_train,
            as_IOB=as_IOB,
            transformer_type=transformer_type,
        ), generate_dataset(
            path=Path(self._path_eval + "\\training\\" + self._task + "\\"),
            output_file=self._output_test,
            as_IOB=as_IOB,
            transformer_type=transformer_type,
        )

    def load_datasets(self, transformer_type="") -> tuple[pd.DataFrame, pd.DataFrame]:
        def load_dataset(
            filename: str, transformer_type="", as_IOB: bool = True
        ) -> pd.DataFrame:
            if transformer_type != "":
                transformer_type = transformer_type.replace("/", "_")
                filename = filename + "_" + transformer_type + ".pkl"
            elif as_IOB:
                filename = filename + "_IOB.pkl"

            try:
                FilesHandler._logger.info(f"Loading DataFrame from: {filename}")
                df: pd.DataFrame = pd.read_pickle(filename)

            except (FileNotFoundError, OSError) as e:
                FilesHandler._logger.error(e)
                return None

            FilesHandler._logger.info("DataFrame succesfully loaded")
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
        if transformer_type != "":
            transformer_type = transformer_type.replace("/", "_")
            train_dataset.to_pickle(
                self._output_train + "_" + transformer_type + ".pkl"
            )
            test_dataset.to_pickle(self._output_test + "_" + transformer_type + ".pkl")
        FilesHandler._logger.info("Datasets successfully saved")

    def load_training_data(
        self,
        features: str,
    ) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:

        try:
            FilesHandler._logger.info(f"Loading training data...")

            X_train = np.load(
                self._output_X_train + "_" + features + ".npy",
                allow_pickle=True,
            )
            X_test = np.load(
                self._output_X_test + "_" + features + ".npy", allow_pickle=True
            )
            y_train = np.load(
                self._output_y_train + "_" + features + ".npy",
                allow_pickle=True,
            )
            y_test = np.load(
                self._output_y_test + "_" + features + ".npy", allow_pickle=True
            )

            FilesHandler._logger.info("Training data succesfully loaded")
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
            np.save(self._output_X_train + "_" + features + ".npy", X_train)
            np.save(self._output_X_test + "_" + features + ".npy", X_test)
            np.save(self._output_y_train + "_" + features + ".npy", y_train)
            np.save(self._output_y_test + "_" + features + ".npy", y_test)

            FilesHandler._logger.info(
                f"Data is successfully saved in dir \\data\\{self._task}"
            )

        except OSError as e:
            FilesHandler._logger.error(e)
            sys.exit()

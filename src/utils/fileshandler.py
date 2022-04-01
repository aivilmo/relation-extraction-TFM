#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np

from logger.logger import Logger
from core.preprocess import Preprocessor


class FilesHandler:

    _logger = Logger.instance()

    @staticmethod
    def generate_datasets(
        path_train: Path,
        path_test: Path,
        output_file_train: str,
        output_file_test: str,
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
                df: pd.DataFrame = Preprocessor.process_content_as_sentences(
                    path, transformer_type=transformer_type
                )
                output_file += "_" + transformer_type + ".pkl"
            elif as_IOB:
                df: pd.DataFrame = Preprocessor.process_content_as_IOB_format(path)
                output_file += "_IOB.pkl"

            # DEPRECATED
            elif as_BILUOV:
                df: pd.DataFrame = Preprocessor.process_content_as_BILUOV_format(path)
                output_file += "_BILUOV.pkl"
            # DEPRECATED
            else:
                output_file += ".pkl"
                df: pd.DataFrame = Preprocessor.process_content(path)

            df.to_pickle(output_file)
            print(df)

            FilesHandler._logger.info(
                f"DataFrame successfully generated and saved at: {output_file}"
            )
            return df

        return generate_dataset(
            path=path_train,
            output_file=output_file_train,
            as_IOB=as_IOB,
            as_BILUOV=as_BILUOV,
            transformer_type=transformer_type,
        ), generate_dataset(
            path=path_test,
            output_file=output_file_test,
            as_IOB=as_IOB,
            as_BILUOV=as_BILUOV,
            transformer_type=transformer_type,
        )

    @staticmethod
    def load_datasets(
        filename_train: str, filename_test: str, transformer_type=""
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        def load_dataset(filename: str, transformer_type="") -> pd.DataFrame:
            filename = filename + "_IOB.pkl"
            if transformer_type != "":
                filename = filename + "_" + transformer_type + ".pkl"

            try:
                FilesHandler._logger.info(f"Loading DataFrame from: {filename}")
                df: pd.DataFrame = pd.read_pickle(filename)
                FilesHandler._logger.info("DataFrame succesfully loaded")
                return df

            except FileNotFoundError as e:
                FilesHandler._logger.error(f'File "{e.filename}" not found')
                return None

        return load_dataset(filename_train, transformer_type), load_dataset(
            filename_test, transformer_type
        )

    @staticmethod
    def load_training_data(
        X_train: str,
        y_train: str,
        X_test: str,
        y_test: str,
        features: str,
    ) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:

        try:
            FilesHandler._logger.info(f"Loading training data...")

            X_train = np.load(X_train + "_" + features + ".npy")
            X_test = np.load(X_test + "_" + features + ".npy")
            y_train = np.load(y_train + "_" + features + ".npy")
            y_test = np.load(y_test + "_" + features + ".npy")

            FilesHandler._logger.info("Training data succesfully loaded")
            return X_train, X_test, y_train, y_test

        except FileNotFoundError as e:
            import sys

            FilesHandler._logger.error(f'File "{e.filename}" not found')
            FilesHandler._logger.error(
                f"Need to generate training files for features: {features}"
            )
            FilesHandler._logger.error(f"Run: 'python .\main.py --generate {features}'")
            sys.exit()

    @staticmethod
    def save_training_data(
        X_train: str,
        y_train: str,
        X_test: str,
        y_test: str,
        features: str,
    ) -> None:
        np.save(X_train[1] + "_" + features + ".npy", X_train[0])
        np.save(X_test[1] + "_" + features + ".npy", y_train[0])
        np.save(y_train[1] + "_" + features + ".npy", X_test[0])
        np.save(y_test[1] + "_" + features + ".npy", y_test[0])

        FilesHandler._logger.info("Data is successfully saved in dir \\data\\")

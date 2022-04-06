#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import numpy as np

from logger.logger import Logger
from core.preprocess import Preprocessor


class FilesHandler:

    _path_ref: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
    _path_eval: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\eval"
    _output_train: str = "data\\ref_train"
    _output_test: str = "data\\eval_train"
    _output_X_train: str = "data\\X_ref_train"
    _output_X_test: str = "data\\X_dev_train"
    _output_y_train: str = "data\\y_ref_train"
    _output_y_test: str = "data\\y_dev_train"

    _logger = Logger.instance()

    @staticmethod
    def generate_datasets(
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
                transformer_type = transformer_type.replace("/", "_")
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
                df: pd.DataFrame = Preprocessor.process_content(
                    path, transformer_type=transformer_type
                )

            df.to_pickle(output_file)
            print(df)

            FilesHandler._logger.info(
                f"DataFrame successfully generated and saved at: {output_file}"
            )
            return df

        return generate_dataset(
            path=Path(FilesHandler._path_ref + "\\training\\"),
            output_file=FilesHandler._output_train,
            as_IOB=as_IOB,
            as_BILUOV=as_BILUOV,
            transformer_type=transformer_type,
        ), generate_dataset(
            path=Path(FilesHandler._path_eval + "\\training\\scenario2-taskA\\"),
            output_file=FilesHandler._output_test,
            as_IOB=as_IOB,
            as_BILUOV=as_BILUOV,
            transformer_type=transformer_type,
        )

    @staticmethod
    def load_datasets(transformer_type="") -> tuple[pd.DataFrame, pd.DataFrame]:
        def load_dataset(filename: str, transformer_type="") -> pd.DataFrame:
            if transformer_type != "":
                transformer_type = transformer_type.replace("/", "_")
                filename = filename + "_" + transformer_type + ".pkl"
            else:
                filename = filename + "_IOB.pkl"

            try:
                FilesHandler._logger.info(f"Loading DataFrame from: {filename}")
                df: pd.DataFrame = pd.read_pickle(filename)
                FilesHandler._logger.info("DataFrame succesfully loaded")
                return df

            except FileNotFoundError as e:
                FilesHandler._logger.error(f'File "{e.filename}" not found')
                return None

        return load_dataset(FilesHandler._output_train, transformer_type), load_dataset(
            FilesHandler._output_test, transformer_type
        )

    @staticmethod
    def load_training_data(
        features: str,
    ) -> tuple[np.ndarray, np.array, np.ndarray, np.array]:

        try:
            FilesHandler._logger.info(f"Loading training data...")

            X_train = np.load(
                FilesHandler._output_X_train + "_" + features + ".npy",
                allow_pickle=True,
            )
            X_test = np.load(
                FilesHandler._output_X_test + "_" + features + ".npy", allow_pickle=True
            )
            y_train = np.load(
                FilesHandler._output_y_train + "_" + features + ".npy",
                allow_pickle=True,
            )
            y_test = np.load(
                FilesHandler._output_y_test + "_" + features + ".npy", allow_pickle=True
            )

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
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        features: np.ndarray,
    ) -> None:
        try:
            np.save(FilesHandler._output_X_train + "_" + features + ".npy", X_train)
            np.save(FilesHandler._output_X_test + "_" + features + ".npy", X_test)
            np.save(FilesHandler._output_y_train + "_" + features + ".npy", y_train)
            np.save(FilesHandler._output_y_test + "_" + features + ".npy", y_test)

            FilesHandler._logger.info("Data is successfully saved in dir \\data\\")

        except OSError as e:
            import sys

            FilesHandler._logger.error(
                f"Cannot save file into a non-existent directory: {e.filename}"
            )
            sys.exit()

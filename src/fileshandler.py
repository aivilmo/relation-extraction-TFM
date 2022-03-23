#!/usr/bin/env python

from pathlib import Path
import pandas as pd

from logger import Logger
from preprocess import Preprocessor


class FilesHandler:
    @staticmethod
    def generate_dataset(
        path: Path,
        output_file: str,
        as_IOB: bool = False,
        as_BILUOV: bool = False,
        as_sentences: bool = True,
        transformer_type: str = "",
    ) -> pd.DataFrame:

        Logger.instance().info("Generating DataFrame...")
        if as_sentences:
            df: pd.DataFrame = Preprocessor.process_content_as_sentences(
                path, transformer_type=transformer_type
            )
        elif as_BILUOV:
            df: pd.DataFrame = Preprocessor.process_content_as_BILUOV_format(path)
        elif as_IOB:
            df: pd.DataFrame = Preprocessor.process_content_as_IOB_format(path)
        else:
            df: pd.DataFrame = Preprocessor.process_content(path)

        df.to_pickle(output_file)

        print(df)
        Logger.instance().info(
            "DataFrame succesfully generated and saved at: " + output_file
        )
        return df

    @staticmethod
    def load_dataset(
        filename: str, transformer_type="", test: bool = False
    ) -> pd.DataFrame:
        Logger.instance().info("Loading DataFrame from: " + filename)
        df: pd.DataFrame = pd.read_pickle(filename)
        Logger.instance().info("DataFrame succesfully loaded")
        if not test:
            df = Preprocessor.data_augmentation(
                df,
                transformer_type=transformer_type,
                last_n_classes=2,
                classes_to_augmentate=["I-Predicate", "I-Action", "B-Reference"],
            )
        return df

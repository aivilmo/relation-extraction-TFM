#!/usr/bin/env python

from pathlib import Path
import pandas as pd

from logger import Logger


class FilesHandler:
    @staticmethod
    def generate_dataset(
        path: Path, output_file: str, as_IOB: bool = True
    ) -> pd.DataFrame:
        from preprocess import Preprocessor

        Logger.instance().info("Generating DataFrame...")

        if as_IOB:
            df: pd.DataFrame = Preprocessor.process_content_as_IOB_format(path)
        else:
            df: pd.DataFrame = Preprocessor.process_content(path)

        df.to_pickle(output_file)

        Logger.instance().info(df)
        Logger.instance().info(
            "DataFrame succesfully generated and saved at: " + output_file
        )
        return df

    @staticmethod
    def load_dataset(filename: str) -> pd.DataFrame:
        Logger.instance().info("Loading DataFrame from: " + filename)
        df: pd.DataFrame = pd.read_pickle(filename)
        Logger.instance().info("DataFrame succesfully loaded")
        return df

#!/usr/bin/env python

from pathlib import Path
import pandas as pd

# pd.set_option("display.max_rows", None)


class FilesHandler:
    @staticmethod
    def generate_dataset(path: Path, output_file: str) -> pd.DataFrame:
        from preprocess import Preprocessor

        print("Generating DataFrame...")

        df: pd.DataFrame = Preprocessor.process_content(path)
        df.to_pickle(output_file)

        print(df)
        print("DataFrame succesfully generated and saved at: " + output_file)
        return df

    @staticmethod
    def load_dataset(filename: str) -> pd.DataFrame:
        print("Loading DataFrame from: " + filename)
        df: pd.DataFrame = pd.read_pickle(filename)
        print("DataFrame succesfully loaded")
        return df

#!/usr/bin/env python

from ehealth.baseline import Baseline
import pandas as pd

# pd.set_option("display.max_rows", None)


class FilesHandler:
    @staticmethod
    def generate_dataset(dir: str, output_file: str) -> pd.DataFrame:
        from preprocess import Preprocessor
        import os

        print("Generating DataFrame...")
        baseline = Baseline()
        _, relations = baseline.fit(dir)
        df = Preprocessor.process_relations(relations)

        text_files: list = [files for _, _, files in os.walk(dir)][0]
        content: str = ""
        for text_file in text_files:
            if not text_file.endswith(".txt"):
                continue

            with open(
                os.path.join(dir, text_file), mode="r", encoding="utf8"
            ) as reader:
                content += reader.read()

        Preprocessor.process_content(df, content)
        df.to_pickle(output_file)
        print(df)
        print("DataFrame succesfully generated and saved at: " + output_file)
        return df

    @staticmethod
    def load_dataset(filename: str) -> pd.DataFrame:
        return pd.read_pickle(filename)

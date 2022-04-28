#!/usr/bin/env python

import pandas as pd

from logger.logger import Logger


class PostProcessor:

    _instance = None
    _dataset = "training"

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if PostProcessor._instance is None:
            PostProcessor()
        return PostProcessor._instance

    def __init__(self, run: int, task: str) -> None:
        if PostProcessor._instance is not None:
            raise Exception

        self._run = "run" + run
        self._task = task

        PostProcessor._instance = self

    def append_row(
        self, df: pd.DataFrame, index: int, words: list, positions: list, tag: str
    ) -> pd.DataFrame:
        entities: str = " ".join(words)
        positions: str = ";".join(positions)

        entity = pd.Series(
            {
                "index": "T" + str(index),
                "entity_pos": tag + " " + positions,
                "word": entities,
            },
            name=index,
        )
        df = df.append(entity)
        return df

    def export_data_to_file(self, dataset_test: pd.DataFrame) -> None:
        df: pd.DataFrame = pd.DataFrame()

        dataset_test = dataset_test[dataset_test.predicted_tag != "O"]
        print(dataset_test)
        dataset_test = dataset_test.append(dataset_test.iloc[-1])

        PostProcessor._logger.info(
            f"Exporting output.ann data for task {self._task}, run {self._run} and dataset {PostProcessor._dataset}"
        )

        index: int = 1
        sentence_offset: int = 0
        pos_end: int = 0

        last_sentence: str = dataset_test.sentence.values[0]
        last_words: list = []
        last_positions: list = []

        last_token: str = dataset_test.original_token.values[0]
        pos_init: int = last_sentence.find(last_token)
        last_pos: str = str(pos_init) + " " + str(pos_init + len(last_token))
        last_tag: str = (
            dataset_test.predicted_tag.values[0].replace("I-", "").replace("B-", "")
        )

        for i, row in dataset_test.iterrows():
            if i == 0:
                continue
            if row.sentence != last_sentence:
                sentence_offset += len(last_sentence) + 1
                pos_end = 0

            tag: str = row.predicted_tag.replace("I-", "").replace("B-", "")
            pos_init: int = (
                row.sentence.find(row.original_token, pos_end) + sentence_offset
            )

            pos_end = pos_init + len(row.original_token)
            pos: str = str(pos_init) + " " + str(pos_end)
            pos_end = pos_end - sentence_offset

            # If the last part was fist part the current part too, we clean our parts
            current_is_first_part: bool = "B-" in row.predicted_tag
            if not current_is_first_part:
                last_words.append(row.original_token)
                last_positions.append(pos)
                continue

            last_words.insert(0, last_token)
            last_positions.insert(0, last_pos)

            df = self.append_row(df, index, last_words, last_positions, last_tag)
            index += 1

            last_token = row.original_token
            last_pos = pos
            last_tag = tag
            last_sentence = row.sentence
            last_words, last_positions = [], []

        print(df)

        PostProcessor._logger.info("Output data successfully exported")
        self.save_output_file(df)

    def save_output_file(self, df: pd.DataFrame) -> None:
        from pathlib import Path

        output_dir = Path(
            "C:\\Users\\Aitana V\\Desktop\\UNIVERSIDAD\\UNED\\TFM\\relation-extraction-TFM\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\submissions\\baseline\\"
            + PostProcessor._dataset
            + "\\"
            + self._run
            + "\\"
            + self._task
            + "\\"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            str(output_dir) + "\\output.ann",
            header=None,
            index=None,
            sep="\t",
            mode="w",
        )

        PostProcessor._logger.info(f"File output.ann saved at path {str(output_dir)}")

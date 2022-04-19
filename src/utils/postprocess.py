#!/usr/bin/env python

import imp
import pandas as pd

from logger.logger import Logger


class PostProcessor:

    _instance = None
    _run = "run1"
    _task = "scenario2-taskA"
    _dataset = "training"

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if PostProcessor._instance is None:
            PostProcessor()
        return PostProcessor._instance

    def __init__(self) -> None:
        if PostProcessor._instance is not None:
            raise Exception

        PostProcessor._instance = self

    @staticmethod
    def export_data_to_file(dataset_test: pd.DataFrame) -> None:
        df: pd.DataFrame = pd.DataFrame()

        dataset_test = dataset_test[dataset_test.predicted_tag != "O"]
        # print(dataset_test[0:55])

        PostProcessor._logger.info(
            f"Exporting output.ann data for task {PostProcessor._task}, run {PostProcessor._run} and dataset {PostProcessor._dataset}"
        )

        index: int = 1
        sentence_offset: int = 0
        last_is_fist_part: bool = False
        sentences: list = [dataset_test.sentence.values[0]]
        last_words: list = []
        last_pos: list = []
        last_position: int = 0
        for _, row in dataset_test.iterrows():
            if row.sentence not in sentences:
                sentences.append(row.sentence)
                sentence_offset += len(row.sentence)
            tag = row.predicted_tag.replace("I-", "").replace("B-", "")
            pos_init = (
                row.sentence.find(row.original_token, last_position) + sentence_offset
            )
            last_position = pos_init + len(row.original_token)
            pos = str(pos_init) + " " + str(last_position)

            last_words.append(row.original_token)
            last_pos.append(pos)

            entities = " ".join(last_words)
            positions = ";".join(last_pos)

            # If the last part was fist part the current part too, we clean our parts
            current_is_first_part = "B-" in row.predicted_tag
            if not last_is_fist_part and not current_is_first_part:
                last_is_fist_part = current_is_first_part
                continue

            if current_is_first_part:
                last_words = [row.original_token]
                last_pos = [pos]
                entities = row.original_token
                positions = pos

            last_is_fist_part = current_is_first_part
            entity = pd.Series(
                {
                    "index": "T" + str(index),
                    "entity_pos": tag + " " + positions,
                    "word": entities,
                },
                name=index,
            )
            index += 1
            df = df.append(entity)
            # print(df[0:40])

        PostProcessor._logger.info("Output data successfully exported")
        PostProcessor.save_output_file(df)

    @staticmethod
    def save_output_file(df: pd.DataFrame) -> None:
        from pathlib import Path

        output_dir = Path(
            "C:\\Users\\Aitana V\\Desktop\\UNIVERSIDAD\\UNED\\TFM\\relation-extraction-TFM\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\submissions\\baseline\\"
            + PostProcessor._dataset
            + "\\"
            + PostProcessor._run
            + "\\"
            + PostProcessor._task
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

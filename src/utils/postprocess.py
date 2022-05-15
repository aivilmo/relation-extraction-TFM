#!/usr/bin/env python

from matplotlib.cbook import strip_math
import pandas as pd

from logger.logger import Logger


class PostProcessor:

    _instance = None
    _dataset = "training"
    _path = (
        "C:\\Users\\Aitana V\\Desktop\\UNIVERSIDAD\\UNED\\TFM\\relation-extraction-TFM\\dataset\\"
        + "ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\submissions\\baseline\\"
    )
    _output_file = "output.ann"

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

    def append_entity_row(
        self, df: pd.DataFrame, index: int, words: list, positions: list, tag: str
    ) -> pd.DataFrame:
        entities: str = " ".join(words)
        pos: str = ";".join(positions)
        entity_pos: str = tag + " " + pos

        if "word" in list(df.columns) and entity_pos in df.entity_pos.unique():
            return df, False

        entity = pd.Series(
            {
                "index": "T" + str(index),
                "entity_pos": entity_pos,
                "word": entities,
            },
            name=index,
        )
        df = df.append(entity)
        return df, True

    def append_entity_relation(
        self, df: pd.DataFrame, index: int, relation_type: str, T1: str, T2: str
    ) -> pd.DataFrame:
        relation = pd.Series(
            {
                "index": "R" + str(index),
                "entity_pos": relation_type,
                "word": "Arg1:" + "T" + str(T1) + " Arg2:" + "T" + str(T2),
            },
            name=index,
        )
        df = df.append(relation)
        return df

    def get_pos(self, sentence: str, token: str, offset: int) -> str:
        positions = []
        for t in token.split():
            pos_init1: int = sentence.find(t) + offset
            pos_end1 = pos_init1 + len(t)
            positions.append(str(pos_init1) + " " + str(pos_end1))

        return positions

    def export_data_to_file(self, dataset_test: pd.DataFrame) -> None:
        df: pd.DataFrame = pd.DataFrame()

        if "predicted_tag" not in list(dataset_test.columns):
            self._logger.error(f"Dataset must be trained with any model before export")
            return

        self._logger.info(
            f"Exporting output.ann data for task {self._task}, run {self._run} and dataset {self._dataset}"
        )

        if "taskA" in self._task:
            dataset_test = dataset_test[dataset_test.predicted_tag != "O"]
            dataset_test = dataset_test.append(dataset_test.iloc[-1])
            df = self.export_taskA(dataset_test)
        if "taskB" in self._task:
            df = self.export_taskB(dataset_test)

        self._logger.info("Output data successfully exported")
        self.save_output_file(df)

    def export_taskA(self, dataset_test: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()

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

            df, _ = self.append_entity_row(
                df, index, last_words, last_positions, last_tag
            )
            index += 1

            last_token = row.original_token
            last_pos = pos
            last_tag = tag
            last_sentence = row.sentence
            last_words, last_positions = [], []

        return df

    def export_taskB(self, dataset_test: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()

        entity_index: int = 1
        relation_index: int = 0
        sentence_offset: int = 0

        relations: list = []

        for sent in dataset_test.sentence.unique():
            sent_df = dataset_test.loc[dataset_test.sentence == sent]
            sent_df = sent_df[sent_df.predicted_tag != "O"]

            # Fill entities
            for _, row in sent_df.iterrows():
                tag1: str = row.tag1.replace("B-", "").replace("I-", "")
                tag2: str = row.tag2.replace("B-", "").replace("I-", "")

                pos1 = self.get_pos(row.sentence, row.original_token1, sentence_offset)
                pos2 = self.get_pos(row.sentence, row.original_token2, sentence_offset)

                df, has_inserted = self.append_entity_row(
                    df, entity_index, [row.original_token1], pos1, tag1
                )
                if has_inserted:
                    entity_index += 1
                df, has_inserted = self.append_entity_row(
                    df, entity_index, [row.original_token2], pos2, tag2
                )
                if has_inserted:
                    entity_index += 1

                T1 = df.loc[df.word == row.original_token1]["index"].values[0][1:]
                T2 = df.loc[df.word == row.original_token2]["index"].values[0][1:]
                relations.append((T1, T2))

            # Fill relations
            for _, row in sent_df.iterrows():
                df = self.append_entity_relation(
                    df,
                    relation_index,
                    row.predicted_tag,
                    relations[relation_index][0],
                    relations[relation_index][1],
                )
                relation_index += 1

            sentence_offset += len(sent) + 1

        print(df)
        return df

    def save_output_file(self, df: pd.DataFrame) -> None:
        from pathlib import Path

        output_dir = Path(
            self._path + self._dataset + "\\" + self._run + "\\" + self._task + "\\"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        df.to_csv(
            str(output_dir) + "\\" + self._output_file,
            header=None,
            index=None,
            sep="\t",
            mode="w",
        )

        self._logger.info(f"File output.ann saved at path {str(output_dir)}")

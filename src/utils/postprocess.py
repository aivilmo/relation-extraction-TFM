#!/usr/bin/env python

import pandas as pd
import re

from logger.logger import Logger
from utils.appconstants import AppConstants


class PostProcessor:

    _instance = None
    _dataset = AppConstants.instance()._test_dataset
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

    def __init__(self) -> None:
        from utils.appconstants import AppConstants

        if PostProcessor._instance is not None:
            raise Exception

        self._run = "run" + AppConstants.instance()._run
        self._task = AppConstants.instance()._task

        PostProcessor._instance = self

    def append_entity_row(
        self, df: pd.DataFrame, index: int, words: list, positions: list, tag: str
    ) -> pd.DataFrame:
        entities: str = " ".join(words)
        entities = self.trim(entities)
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

    def trim(self, phrase: str) -> str:
        return (
            phrase.replace(".", "")
            .replace(",", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
        )

    def get_symbols(self, phrase: str, offset: int) -> list:
        symbols: list = []
        symbols += [_.start() + offset for _ in re.finditer("\.", phrase)]
        symbols += [_.start() + offset for _ in re.finditer(",", phrase)]
        return symbols

    def get_pos(self, sentence: str, token: str, offset: int) -> str:
        positions = []
        # For avoid no whole words
        symbols = self.get_symbols(sentence, offset)
        sentence = " " + self.trim(sentence) + " "
        token = self.trim(token)

        for t in token.split():
            pos_init1: int = sentence.find(" " + t + " ") + offset
            for s in symbols:
                if s < pos_init1:
                    pos_init1 += 1
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
            df = self.export_taskA(dataset_test)
        if "taskB" in self._task:
            df = self.export_taskB(dataset_test)

        self._logger.info("Output data successfully exported")
        self.save_output_file(df)

    def export_taskA(self, dataset_test: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()

        sentence_offset: int = 0
        index: int = 1
        # sent = "La terapia dirigida es un tipo de tratamiento en el que se utilizan sustancias para identificar y atacar células cancerosas específicas sin dañar las células normales."
        sentences = list(dataset_test.sentence.unique())
        # sentences.insert(38, sent)

        for sent in sentences:
            sent_df = dataset_test.loc[dataset_test.sentence == sent]
            sent_df = sent_df[sent_df.predicted_tag != "O"]

            if len(sent_df) == 0:
                continue

            first = sent_df.iloc[0]
            entities = [first.original_token]
            positions = self.get_pos(
                first.sentence, first.original_token, sentence_offset
            )
            last_tag = first.predicted_tag.replace("B-", "").replace("I-", "")

            for _, row in sent_df.iterrows():
                tag: str = row.predicted_tag.replace("B-", "").replace("I-", "")
                pos = self.get_pos(row.sentence, row.original_token, sentence_offset)
                if pos == []:
                    continue
                pos = pos[0]

                if row.predicted_tag.startswith("B-"):
                    df, has_inserted = self.append_entity_row(
                        df, index, entities, positions, last_tag
                    )
                    entities = [row.original_token]
                    positions = [pos]
                    last_tag = tag
                    if has_inserted:
                        index += 1
                    continue

                entities.append(row.original_token)
                positions.append(pos)

            sentence_offset += len(sent) + 1

        return df

    def export_taskB(self, dataset_test: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()

        entity_index: int = 1
        relation_index: int = 0
        sentence_offset: int = 0

        relations: list = []
        max_sentences = len(dataset_test.sentence.unique())
        i = 0
        for sent in dataset_test.sentence.unique():
            sent_df = dataset_test.loc[dataset_test.sentence == sent]
            # Avoid create non testing sentences
            if len(sent_df[sent_df.tag != "O"]) == 0:
                sent_df = sent_df[sent_df.tag != "O"]

            if "taskB" in self._task:
                sent_df = sent_df[sent_df.predicted_tag != "O"]
            sent_df = sent_df[sent_df.tag1 != "O"]
            sent_df = sent_df[sent_df.tag2 != "O"]
            sent_df = sent_df[sent_df.original_token1 != "."]
            sent_df = sent_df[sent_df.original_token2 != "."]
            sent_df = sent_df[sent_df.original_token1 != ","]
            sent_df = sent_df[sent_df.original_token2 != ","]
            sent_df = sent_df[sent_df.original_token1 != ")"]
            sent_df = sent_df[sent_df.original_token2 != ")"]
            sent_df = sent_df[sent_df.original_token1 != "("]
            sent_df = sent_df[sent_df.original_token2 != "("]
            sent_df = sent_df[sent_df.original_token1 != "()"]
            sent_df = sent_df[sent_df.original_token2 != "()"]
            sent_df = sent_df[sent_df.original_token1 != "(),"]
            sent_df = sent_df[sent_df.original_token2 != "(),"]
            sent_df = sent_df[sent_df.original_token1 != "),"]
            sent_df = sent_df[sent_df.original_token2 != "),"]

            sent_df = sent_df[sent_df.original_token1 != "(,"]
            sent_df = sent_df[sent_df.original_token2 != "(,"]
            sent_df = sent_df[sent_df.original_token1 != '""']
            sent_df = sent_df[sent_df.original_token2 != '""']

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

                token1 = self.trim(row.original_token1)
                T1 = df.loc[df.word == token1]["index"].values[-1][1:]

                token2 = self.trim(row.original_token2)
                T2 = df.loc[df.word == token2]["index"].values[-1][1:]

                relations.append((T1, T2))

            # Fill relations
            for _, row in sent_df.iterrows():
                if row.tag == "O":
                    continue
                df = self.append_entity_relation(
                    df,
                    relation_index,
                    row.predicted_tag,
                    relations[relation_index][0],
                    relations[relation_index][1],
                )
                relation_index += 1

            sentence_offset += len(sent) + 1
            if i % 100 == 0:
                self._logger.info(f"Finished with {i} of {max_sentences} sentences")
            i += 1
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

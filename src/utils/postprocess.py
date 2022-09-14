#!/usr/bin/env python

import pandas as pd
from ehealth.anntools import Collection, Sentence, Relation, Keyphrase
from pathlib import Path

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
        if PostProcessor._instance is not None:
            raise Exception

        self._run = "run" + AppConstants.instance()._run
        self._task = AppConstants.instance()._task

        PostProcessor._instance = self

    def export_data_to_file(self, dataset_test: pd.DataFrame) -> None:
        if "predicted_tag" not in list(dataset_test.columns):
            self._logger.error(f"Dataset must be trained with any model before export")
            return

        self._logger.info(
            f"Exporting output.ann data for task {self._task}, run {self._run} and dataset {self._dataset}"
        )

        if "taskA" in self._task:
            collection = self.export_taskA(dataset_test)

        if "taskB" in self._task:
            collection = self.export_taskB(dataset_test)

        self.save_output_file(collection)
        self._logger.info("Output data successfully exported")

    def pos_list(self, positions: str) -> str:
        pos = []
        for elems in positions.split(";"):
            init, end = elems.split(":")
            pos += [str(init) + " " + str(end)]
        return pos

    def export_taskA(self, dataset_test: pd.DataFrame) -> Collection:
        index: int = 1
        sentences = list(dataset_test.sentence.unique())
        sentences_list = []

        for sent in sentences:
            sentence_obj = Sentence(text=sent)
            keyphrases = []
            sent_df = dataset_test.loc[dataset_test.sentence == sent]
            sent_df = sent_df[sent_df.predicted_tag != "O"]
            # Reverse df
            sent_df = sent_df[::-1]

            stack_pos = []
            for _, row in sent_df.iterrows():
                if "I-" in row.predicted_tag:
                    stack_pos += self.pos_list(row.positions)
                    continue

                positions = stack_pos + self.pos_list(row.positions)
                label = row.predicted_tag.replace("B-", "")

                position = []
                positions.reverse()
                for pos in positions:
                    init, end = pos.split(" ")
                    position.append((int(init), int(end)))
                keyphrase_obj = Keyphrase(
                    sentence=sentence_obj,
                    spans=position,
                    label=label,
                    id=index,
                )
                keyphrases.append(keyphrase_obj)
                index += 1
                stack_pos = []

            sentence_obj.keyphrases = keyphrases
            sentences_list.append(sentence_obj)

        return Collection(sentences_list)

    def export_taskB(self, dataset_test: pd.DataFrame) -> Collection:
        entity_index: int = 1
        sentences = list(dataset_test.sentence.unique())
        sentences_list = []

        for sent in sentences:
            sentence_obj = Sentence(text=sent)
            keyphrases, relations = [], []
            entities = {}
            sent_df = dataset_test.loc[dataset_test.sentence == sent]
            sent_df = sent_df[sent_df.predicted_tag != "O"]
            sent_df = sent_df[sent_df.tag1 != "O"]
            sent_df = sent_df[sent_df.tag2 != "O"]

            for _, row in sent_df.iterrows():
                from_positions = self.pos_list(row.position1)
                to_positions = self.pos_list(row.position2)

                from_ps = []
                for from_p in from_positions:
                    init, end = from_p.split(" ")
                    from_ps.append((int(init), int(end)))

                keyphrase_obj = Keyphrase(
                    sentence=sentence_obj,
                    spans=from_ps,
                    label=row.tag1,
                    id=entity_index,
                )
                if entities.get(keyphrase_obj.text, 0) == 0:
                    entities[keyphrase_obj.text] = entity_index
                    entity_index += 1
                    keyphrases.append(keyphrase_obj)

                to_ps = []
                for to_p in to_positions:
                    init, end = to_p.split(" ")
                    to_ps.append((int(init), int(end)))

                keyphrase_obj = Keyphrase(
                    sentence=sentence_obj,
                    spans=to_ps,
                    label=row.tag2,
                    id=entity_index,
                )
                if entities.get(keyphrase_obj.text, 0) == 0:
                    entities[keyphrase_obj.text] = entity_index
                    entity_index += 1
                    keyphrases.append(keyphrase_obj)

                relation_obj = Relation(
                    sentence=sentence_obj,
                    origin=entities[row.original_token1],
                    destination=entities[row.original_token2],
                    label=row.predicted_tag,
                )
                relations.append(relation_obj)

            sentence_obj.keyphrases = keyphrases
            sentence_obj.relations = relations
            sentences_list.append(sentence_obj)

        return Collection(sentences_list)

    def save_output_file(self, collection: Collection) -> None:
        output_dir = Path(
            self._path + self._dataset + "\\" + self._run + "\\" + self._task + "\\"
        )
        output_dir.mkdir(parents=True, exist_ok=True)

        collection.dump(Path(str(output_dir) + "\\" + self._output_file))

        self._logger.info(f"File output.ann saved at path {str(output_dir)}")

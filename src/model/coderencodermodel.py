#!/usr/bin/env python

import pandas as pd

from model.abstractmodel import AbstractModel


class CoderEncoderModel(AbstractModel):

    _instance = None

    @staticmethod
    def instance():
        if CoderEncoderModel._instance is None:
            CoderEncoderModel()
        return CoderEncoderModel._instance

    def __init__(self) -> None:
        super().__init__()
        if CoderEncoderModel._instance is not None:
            raise Exception

        self._train_data = None
        self._test_data = None

        CoderEncoderModel._instance = self

    @classmethod
    def start_training(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
    ) -> None:
        train_df.rename(columns={"word": "words", "tag": "labels"}, inplace=True)
        test_df.rename(columns={"word": "words", "tag": "labels"}, inplace=True)
        self._labels = list(train_df.labels.unique())

        self.build(train_df, test_df, model="roberta")
        self.train()
        self.evaluate()

    @classmethod
    def build(self, train_df: pd.DataFrame, test_df: pd.DataFrame, model: str) -> None:
        from simpletransformers.ner import NERArgs, NERModel

        self._train_data = train_df
        self._test_data = test_df

        AbstractModel._logger.info(f"Building model {model}...")

        model_args = NERArgs()
        model_args.train_batch_size = 16
        model_args.evaluate_during_training = False
        model_args.labels_list = self._labels
        model_args.save_best_model = True
        model_args.classification_report = True

        self._model = NERModel(
            model_type=model,
            model_name="outputs",
            args=model_args,
            use_cuda=False,
        )
        AbstractModel._logger.info(f"Model has built successfully")

    @classmethod
    def train(self) -> None:
        from time import time

        self._logger.info("Training model...")
        start: float = time()

        self._model.train_model(train_data=self._train_data)

        end: float = time() - start
        self._logger.info(f"Model trained, time: {round(end / 60, 2)} minutes")

    @classmethod
    def evaluate(self) -> None:
        self._logger.info(f"Testing model {self._model}")

        result, model_outputs, preds_list = self._model.eval_model(self._test_data)
        self._logger.info(result)

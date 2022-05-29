#!/usr/bin/env python
import numpy as np
from sklearn.preprocessing import LabelEncoder

from model.abstractmodel import AbstractModel


class CoderEncoderModel(AbstractModel):

    _instance = None

    @staticmethod
    def instance():
        if CoderEncoderModel._instance is None:
            CoderEncoderModel()
        return CoderEncoderModel._instance

    def __init__(self, train, test) -> None:
        super().__init__()
        if CoderEncoderModel._instance is not None:
            raise Exception

        self._train_data = train
        self._test_data = test

        CoderEncoderModel._instance = self

    def start_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model,
    ) -> None:
        self._train_data.rename(
            columns={"token": "words", "tag": "labels"}, inplace=True
        )
        self._test_data.rename(
            columns={"token": "words", "tag": "labels"}, inplace=True
        )
        le = LabelEncoder()
        self._train_data["sentence_id"] = le.fit_transform(self._train_data.sentence)
        self._test_data["sentence_id"] = le.transform(self._test_data.sentence)

        self._labels = list(self._train_data.labels.unique())

        self.build()
        self.train()
        self.evaluate()
        self.export_results()

    def build(self) -> None:
        from simpletransformers.ner import NERArgs, NERModel

        AbstractModel._logger.info("Building model bert-base-multilingual-cased...")

        model_args = NERArgs()
        model_args.train_batch_size = 16
        model_args.evaluate_during_training = False
        model_args.labels_list = self._labels
        model_args.save_best_model = True
        model_args.classification_report = True
        model_args.num_train_epochs = 10
        model_args.use_early_stopping = True
        model_args.early_stopping_patience = 3
        model_args.n_gpu = 8
        model_args.overwrite_output_dir = True

        self._model = NERModel(
            model_type="bert",
            model_name="bert-base-multilingual-cased",
            args=model_args,
            use_cuda=False,
        )
        AbstractModel._logger.info(f"Model has built successfully")

    def train(self) -> None:
        from time import time

        self._logger.info("Training model...")
        start: float = time()

        self._model.train_model(train_data=self._train_data)

        end: float = time() - start
        self._logger.info(f"Model trained, time: {round(end / 60, 2)} minutes")

    def evaluate(self) -> None:
        self._logger.info(f"Testing model {self._model}")
        self._model.eval_model(self._test_data)

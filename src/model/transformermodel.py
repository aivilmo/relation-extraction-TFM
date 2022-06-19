#!/usr/bin/env python
import numpy as np
import datasets
import pandas as pd
from datasets import load_metric

from model.abstractmodel import AbstractModel
from core.embeddinghandler import TransformerEmbedding
import sys


class TransformerModel(AbstractModel):

    _instance = None

    _epochs = 15

    @staticmethod
    def instance():
        if TransformerModel._instance is None:
            TransformerModel()
        return TransformerModel._instance

    def __init__(self, train, test) -> None:
        from sklearn.preprocessing import LabelEncoder
        from utils.appconstants import AppConstants

        super().__init__()
        if TransformerModel._instance is not None:
            raise Exception

        self._task = AppConstants.instance()._task
        self._save_path = "data\\" + self._task + "\\models\\transformer_"

        self._labels = list(train.tag.unique())
        self._le: LabelEncoder = LabelEncoder()
        self._le.fit(self._labels)

        self._dataset_train = datasets.Dataset.from_dict(
            self.parse_dataframe(train), split="train"
        )
        self._dataset_test = datasets.Dataset.from_dict(
            self.parse_dataframe(test), split="test"
        )
        self._model_name = AppConstants.instance()._features[0]
        self._model = None
        self._is_trained = True
        self._args = None

        TransformerModel._instance = self

    def start_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model,
    ) -> None:
        self.build()
        if not self._is_trained:
            self.train()
        self.evaluate()
        self.export_results()

    def build(self) -> None:
        from transformers import (
            TrainingArguments,
            Trainer,
            DataCollatorForTokenClassification,
        )

        self._logger.info(f"Building model {self._model_name}")
        tr_instance = TransformerEmbedding.instance()

        if not tr_instance.trained():
            tr_instance.build_transformer_to_finetuning(
                self._model_name, len(self._labels)
            )

        self.load_model()
        if self._model == None:
            self._model = tr_instance._encoder_layer
            self._is_trained = False

        self._X = self._dataset_train.map(self.tokenize_and_align_labels)
        self._y = self._dataset_test.map(self.tokenize_and_align_labels)

        self._args = TrainingArguments(
            output_dir=f"{self._model_name}-finetuned-{self._task}",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            num_train_epochs=self._epochs,
            weight_decay=0.01,
            push_to_hub=False,
        )

        data_collator = DataCollatorForTokenClassification(
            TransformerEmbedding.instance()._preprocess_layer
        )
        self._model = Trainer(
            self._model,
            self._args,
            train_dataset=self._X,
            eval_dataset=self._y,
            data_collator=data_collator,
            tokenizer=TransformerEmbedding.instance()._preprocess_layer,
            compute_metrics=self.compute_metrics,
        )

        self._logger.info(f"Model {self._model_name} has built")

    def train(self) -> None:
        self._logger.info(f"Training transformer model...")

        self._model.train()

        path = ".\\" + self._save_path + self._model_name
        self._model.save_model(path)
        self._logger.info(f"Model has successfully trained and saved at {path}")

    def evaluate(self) -> None:
        self._logger.info(f"Evaluating transformer model...")

        self._model.evaluate()

        predictions, labels, _ = self._model.predict(self._y)
        predictions = np.argmax(predictions, axis=2)
        y = self.get_tokens_predictions(labels)
        yhat = self.get_tokens_predictions(predictions)

        super().evaluate(yhat, y)

    def load_model(self) -> None:
        from transformers import AutoModelForTokenClassification

        path = ".\\" + self._save_path + self._model_name
        try:
            self._model = AutoModelForTokenClassification.from_pretrained(path)
            self._logger.info(f"Model loaded successfully from path {path}")
        except Exception as e:
            self._logger.warning(e)
            self._logger.warning(f"Model not found in path {path}, we must to train it")

    def parse_dataframe(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        df: pd.DataFrame = pd.DataFrame()
        index: int = 0
        for sent in dataframe.sentence.unique():
            sent_df = dataframe.loc[dataframe.sentence == sent]
            sentence = pd.Series(
                {
                    "tokens": sent_df.original_token.values,
                    "ner_tags": self._le.transform(sent_df.tag.values),
                },
                name=index,
            )

            index += 1
            df = df.append(sentence)
        return df

    def tokenize_and_align_labels(self, dataset: datasets.Dataset) -> datasets.Dataset:
        tokenized_inputs = TransformerEmbedding.instance()._preprocess_layer(
            dataset["tokens"], truncation=True, is_split_into_words=True
        )

        label = dataset["ner_tags"]
        word_ids = tokenized_inputs.word_ids()
        label_ids = []
        for word_idx in word_ids:
            tag = -100
            if word_idx is not None:
                tag = label[word_idx]
            label_ids.append(tag)

        tokenized_inputs["labels"] = label_ids
        tokenized_inputs["word_idx"] = word_ids
        return tokenized_inputs

    def get_tokens_predictions(self, labels: list) -> list:
        global_preds = []
        for sent in range(len(self._y)):
            sentence = self._y[sent]
            labels_sent = labels[sent][1 : len(sentence["word_idx"]) - 1]
            # Remove first and last padding
            idx_sent = sentence["word_idx"][1:-1]

            idxs = []
            count = -1
            for idx in idx_sent:
                count += 1
                if idx in idxs:
                    continue
                global_preds.append(labels_sent[count])
                idxs.append(idx)

        return global_preds

    def compute_metrics(self, p):
        metric = load_metric("seqeval")

        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        true_predictions = [
            [self._labels[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self._labels[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = metric.compute(predictions=true_predictions, references=true_labels)
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }

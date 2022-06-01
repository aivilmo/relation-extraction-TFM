#!/usr/bin/env python
import numpy as np
import datasets

from model.abstractmodel import AbstractModel
from core.embeddinghandler import TransformerEmbedding


class TransformerModel(AbstractModel):

    _instance = None

    _epochs = 1

    @staticmethod
    def instance():
        if TransformerModel._instance is None:
            TransformerModel()
        return TransformerModel._instance

    def __init__(self, train, test) -> None:
        super().__init__()
        if TransformerModel._instance is not None:
            raise Exception

        self._labels = list(train["tag"].unique())
        self._dataset_train = datasets.Dataset.from_dict(train, split="train")
        self._dataset_test = datasets.Dataset.from_dict(test, split="test")
        self._optimizer = None
        self._learning_rate_scheduler = None
        self._training_steps = None

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
        self.train()
        self.evaluate()
        # self.export_results()

    def build(self) -> None:
        from transformers import get_scheduler
        from torch.optim import AdamW
        from main import Main

        feat = Main.instance().features()[0]

        self._logger.info(f"Building model {feat}")

        if not TransformerEmbedding.trained():
            TransformerEmbedding.instance().build_transformer_to_finetuning(
                feat, len(self._labels)
            )

        self._model = TransformerEmbedding.instance()._encoder_layer
        self._X = self.get_data_loader(self._dataset_train)
        self._y = self.get_data_loader(self._dataset_test)
        self._optimizer = AdamW(self._model.parameters(), lr=5e-5)
        self._training_steps = self._epochs * len(self._X)
        self._learning_rate_scheduler = get_scheduler(
            name="linear",
            optimizer=self._optimizer,
            num_warmup_steps=0,
            num_training_steps=self._training_steps,
        )

        self._logger.info(
            f"Model {feat} has built with optimizer {self._optimizer}, training steps {self._training_steps}"
        )

    def train(self) -> None:
        from tqdm.auto import tqdm

        self._logger.info(f"Training transformer model...")
        progress_bar = tqdm(range(self._training_steps))

        self._model.train()
        for _ in range(self._epochs):
            for batch in self._X:
                batch = {k: v for k, v in batch.items()}
                outputs = self._model(**batch)
                loss = outputs.loss
                loss.backward()

                self._optimizer.step()
                self._learning_rate_scheduler.step()
                self._optimizer.zero_grad()
                progress_bar.update(1)

            self._logger.info(f"Model has successfully trained")

    def evaluate(self) -> None:
        from tqdm.auto import tqdm
        from torch import no_grad, argmax

        self._logger.info(f"Evaluating transformer model...")
        progress_bar = tqdm(range(len(self._y)))

        accuracy = datasets.load_metric("accuracy")
        precision = datasets.load_metric("precision")
        recall = datasets.load_metric("recall")
        f1score = datasets.load_metric("f1")

        self._model.eval()
        for batch in self._y:
            batch = {k: v for k, v in batch.items()}
            with no_grad():
                outputs = self._model(**batch)

            logits = outputs.logits
            predictions = argmax(logits, dim=-1)

            accuracy.add_batch(predictions=predictions, references=batch["labels"])
            precision.add_batch(predictions=predictions, references=batch["labels"])
            recall.add_batch(predictions=predictions, references=batch["labels"])
            f1score.add_batch(predictions=predictions, references=batch["labels"])

            progress_bar.update(1)

        self._logger.info(f"Evaluation: ")
        self._logger.info(accuracy.compute())
        self._logger.info(precision.compute(average="macro"))
        self._logger.info(recall.compute(average="macro"))
        self._logger.info(f1score.compute(average="macro"))

    def tokenize(self, dataset):
        labels = datasets.ClassLabel(num_classes=len(self._labels), names=self._labels)

        def tokenize_function(dataset):
            tokens = TransformerEmbedding.instance()._preprocess_layer(
                dataset["token"], padding="max_length", truncation=True
            )
            tokens["labels"] = labels.str2int(dataset["tag"])
            return tokens

        tokenized_datasets = dataset.map(tokenize_function, batched=True)
        for columns in dataset.features:
            tokenized_datasets = tokenized_datasets.remove_columns([columns])
        tokenized_datasets.set_format("torch")
        return tokenized_datasets

    def get_data_loader(self, dataset, batch_size=24):
        from torch.utils.data import DataLoader

        self._logger.info("Loading dataloader from dataset {dataset}")
        tokenized_datasets = self.tokenize(dataset)
        tokenized_datasets = tokenized_datasets.shuffle(seed=42).select(range(10))
        return DataLoader(tokenized_datasets, batch_size=batch_size)

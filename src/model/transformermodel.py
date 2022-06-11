#!/usr/bin/env python
import numpy as np
import datasets

from model.abstractmodel import AbstractModel
from core.embeddinghandler import TransformerEmbedding


class TransformerModel(AbstractModel):

    _instance = None

    _epochs = 1
    _save_path = "data\\models\\transformer_"

    @staticmethod
    def instance():
        if TransformerModel._instance is None:
            TransformerModel()
        return TransformerModel._instance

    def __init__(self, train, test) -> None:
        from main import Main

        super().__init__()
        if TransformerModel._instance is not None:
            raise Exception

        self._labels = list(train.tag.unique())
        self._dataset_train = datasets.Dataset.from_dict(train, split="train")
        self._dataset_test = datasets.Dataset.from_dict(test, split="test")
        self._optimizer = None
        self._learning_rate_scheduler = None
        self._training_steps = None
        self._model_name = Main.instance().features()[0]
        self._model = None
        self._is_trained = True

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
        from transformers import get_scheduler
        from torch.optim import AdamW

        self._logger.info(f"Building model {self._model_name}")
        tr_instance = TransformerEmbedding.instance()

        self.load_model()
        if self._model == None:
            if not tr_instance.trained():
                tr_instance.build_transformer_to_finetuning(
                    self._model_name, len(self._labels)
                )
            self._model = tr_instance._encoder_layer
            self._is_trained = False

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
            f"Model {self._model_name} has built with optimizer {self._optimizer}, training steps {self._training_steps}"
        )

    def train(self) -> None:
        from tqdm.auto import tqdm
        from torch import save

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
        save(self._model, self._save_path + self._model_name)

    def evaluate(self) -> None:
        from tqdm.auto import tqdm
        from torch import no_grad, argmax, empty, cat

        self._logger.info(f"Evaluating transformer model...")
        progress_bar = tqdm(range(len(self._y)))

        yhat: empty = empty([0])
        y: empty = empty([0])
        self._model.eval()
        for batch in self._y:
            batch = {k: v for k, v in batch.items()}
            with no_grad():
                outputs = self._model(**batch)

            logits = outputs.logits
            predictions = argmax(logits, dim=-1)
            yhat = cat((yhat, predictions), axis=0)
            y = cat((y, batch["labels"]), axis=0)
            progress_bar.update(1)

        yhat = cat([yhat[:0], yhat[1:]]).numpy()
        y = cat([y[:0], y[1:]]).numpy()
        super().evaluate(yhat, y)

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

    def get_data_loader(self, dataset, batch_size=20):
        from torch.utils.data import DataLoader

        self._logger.info("Loading dataloader from dataset {dataset}")
        tokenized_datasets = self.tokenize(dataset)
        tokenized_datasets = tokenized_datasets.shuffle(seed=42)  # .select(range(48))
        return DataLoader(tokenized_datasets, batch_size=batch_size)

    def load_model(self) -> None:
        from torch import load

        try:
            path = self._save_path + self._model_name
            self._model = load(path)
            self._logger.info(f"Model loaded successfully from path {path}")
        except:
            self._logger.info("Model has not found, we must to train it")

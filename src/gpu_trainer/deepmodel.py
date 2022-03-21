#!/usr/bin/env python

import random
from typing import Iterator
import numpy as np
import tensorflow as tf

seed = 6
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)


# from os import environ

# environ["KERAS_BACKEND"] = "plaidml.keras.backend"


class DeepModel:
    _instance = None
    _epochs = 80
    _batch_size = 64
    _n_classes = None
    _embedding_dims = 100

    @staticmethod
    def instance():
        if DeepModel._instance == None:
            DeepModel()
        return DeepModel._instance

    def __init__(self) -> None:
        if DeepModel._instance != None:
            raise Exception
        self._model = None
        self._loss = None
        self._history = None
        self._X = None
        self._y = None
        DeepModel._instance = self

    def create_simple_NN(
        self,
        # hidden_layers: int = 5,
        # num_units: list = [768, 640, 512, 384, 256, 128],
        hidden_layers: int = 0,
        num_units: list = [128],
        activation: str = "relu",
        optimizer: str = "adam",
    ):
        self._model = tf.keras.models.Sequential()

        # Input layer
        self._model.add(
            tf.keras.layers.Dense(
                units=num_units[0],
                activation=activation,
                input_shape=(None, self._X.shape[1]),
            )
        )
        self._model.add(tf.keras.layers.Dropout(0.25))

        #  Hidden layers
        for hl in range(hidden_layers):
            self._model.add(
                tf.keras.layers.Dense(
                    units=num_units[hl + 1],
                    activation=activation,
                )
            )
            self._model.add(tf.keras.layers.Dropout(0.25))

        # Output layer
        self._model.add(
            tf.keras.layers.Dense(units=DeepModel._n_classes, activation="softmax")
        )

        self._model.compile(
            loss=self._loss,
            optimizer=optimizer,
            metrics=["accuracy"],
        )

        print(self._model.summary())

    def create_GRU_RNN(self, vocab_size: int, input_length: int):
        self._model = tf.keras.models.Sequential()

        self._model = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=DeepModel._embedding_dims,
            input_length=input_length,
        )

        self._model.add(
            tf.keras.layers.GRU(
                units=DeepModel._embedding_dims,
                dropout=0.1,
                recurrent_dropout=0.5,
            )
        )
        self._model.add(tf.keras.layers.Dense(units=DeepModel._n_classes))

        self._model.compile(
            loss="binary_crossentropy",
            optimizer="rmsprop",
            metrics=["accuracy"],
        )

        print(self._model.summary())

    def train_NN(self):
        self._history = self._model.fit(
            self._X,
            self._y,
            epochs=DeepModel._epochs,
            batch_size=DeepModel._batch_size,
            validation_split=0.2,
            callbacks=DeepModel.get_callbacks(),
            class_weight=self.get_class_weight(),
        )

    def evaluate_NN(self, X, y):
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )
        import matplotlib.pyplot as plt

        print("Testing model...")

        y_hat = self._model.predict(X)
        y_hat = np.argmax(y_hat, axis=1)
        y = np.argmax(y, axis=1)

        print("Classification report:")
        print(classification_report(y, y_hat))

        print("Confusion matrix:")
        cm = confusion_matrix(y, y_hat)
        disp = ConfusionMatrixDisplay(
            confusion_matrix=cm, display_labels=np.unique(y_hat)
        )
        disp.plot()

        plt.show()
        # plt.savefig("confusion_matrix.png")

    def get_class_weight(self) -> dict:
        samples = self._y.shape[0]
        unique, counts = np.unique(np.argmax(self._y, axis=1), return_counts=True)
        return dict(zip(unique, 100 - (counts / samples) * 100))

    # SOURCE: https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab
    def get_class_weight_imbalanced(self, beta: float = 0.9) -> np.ndarray:
        unique, samples_per_cls = np.unique(
            np.argmax(self._y, axis=1), return_counts=True
        )
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * DeepModel._n_classes

        return dict(zip(unique, weights))

    @staticmethod
    def get_callbacks() -> list:
        accuracy = tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        )
        loss = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )

        # Create a callback that saves the model's weights
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, save_weights_only=True, verbose=1
        )
        return [accuracy, loss, checkpoint]

    def show_history(self) -> None:
        import matplotlib.pyplot as plt

        acc = self._history.history["accuracy"]
        val_acc = self._history.history["val_accuracy"]
        loss = self._history.history["loss"]
        val_loss = self._history.history["val_loss"]

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, "bo", label="Training acc")
        plt.plot(epochs, val_acc, "b", label="Validation acc")
        plt.title("Training and validation accuracy")
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, "bo", label="Training loss")
        plt.plot(epochs, val_loss, "b", label="Validation loss")
        plt.title("Training and validation loss")
        plt.legend()

        plt.show()
        # plt.savefig("history.png")

    def _load_data(self, features: list) -> None:
        feat = features[0].replace("/", "_")
        self._X, X_test = np.load("..\\data\\X_train_" + feat + ".npy"), np.load(
            "..\\data\\X_test_" + feat + ".npy"
        )
        self._y, y_test = np.load("..\\data\\y_train_" + feat + ".npy"), np.load(
            "..\\data\\y_test_" + feat + ".npy"
        )
        DeepModel._n_classes = self._y.shape[1]
        return X_test, y_test

    def undersample_data(self) -> None:
        from imblearn.under_sampling import NearMiss

        unsersampler = NearMiss(
            n_neighbors=1, n_neighbors_ver3=3, sampling_strategy="majority"
        )
        self._X, self._y = unsersampler.fit_resample(self._X, self._y)

    def oversample_data(self) -> None:
        from imblearn.over_sampling import RandomOverSampler, ADASYN

        # oversampler = RandomOverSampler(sampling_strategy="minority")
        oversampler = ADASYN(
            # sampling_strategy="minority",
            random_state=0,
            n_neighbors=1,
        )
        self._X, self._y = oversampler.fit_resample(self._X, self._y)

    def combined_resample_data(self) -> None:
        from imblearn.combine import SMOTETomek, SMOTEENN

        self._X, self._y = SMOTEENN().fit_resample(self._X, self._y)

    @staticmethod
    def main() -> None:
        from argsparser import ArgsParser
        import tensorflow_addons as tfa

        args = ArgsParser.get_args()
        X_test, y_test = DeepModel.instance()._load_data(args.features)

        if args.loss == None or "binary_crossentropy" in args.loss:
            DeepModel.instance()._loss = tf.keras.losses.BinaryCrossentropy()
        elif "sigmoid_focal_crossentropy" in args.loss:
            DeepModel.instance()._loss = tfa.losses.SigmoidFocalCrossEntropy(
                alpha=0.20, gamma=2.0
            )

        if args.model == None or "basic_nn" in args.model:
            DeepModel.instance().create_simple_NN()
        elif "gru" in args.model:
            X_shape = DeepModel.instance()._X.shape
            DeepModel.instance()._X = DeepModel.instance()._X.reshape(
                X_shape[0], 1, X_shape[1]
            )
            X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
            DeepModel.instance().create_GRU_RNN()
        elif "lstm" in args.model:
            pass

        if "oversampling" in args.imbalance_strategy:
            DeepModel.instance().oversample_data()
        elif "undersampling" in args.imbalance_strategy:
            DeepModel.instance().undersample_data()
        elif "both" in args.imbalance_strategy:
            DeepModel.instance().combined_resample_data()

        DeepModel.instance().train_NN()
        DeepModel.instance().evaluate_NN(X=X_test, y=y_test)
        DeepModel.instance().show_history()


if __name__ == "__main__":
    DeepModel.instance().main()

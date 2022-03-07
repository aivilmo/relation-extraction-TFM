#!/usr/bin/env python

import random
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
    epochs = 80
    batch_size = 32
    n_classes = 8

    @staticmethod
    def instance():
        if DeepModel._instance == None:
            DeepModel()
        return DeepModel._instance

    def __init__(self) -> None:
        if DeepModel._instance != None:
            raise Exception
        self._model = None
        self._history = None
        self._X = None
        self._y = None
        DeepModel._instance = self

    def test_model(self, X, y) -> None:
        from sklearn.metrics import classification_report

        yhat = self._model.predict(X)
        yhat = np.argmax(yhat, axis=1)
        y = np.argmax(y, axis=1)

        print(classification_report(y, yhat))

    def create_simple_NN(
        self,
        hidden_layers: int = 1,
        num_units: list = [128],
        activation: str = "relu",
        optimizer: str = "adam",
        loss: str = "binary_crossentropy",
    ):
        import tensorflow_addons as tfa

        self._model = tf.keras.models.Sequential()

        # Input layer
        self._model.add(
            tf.keras.layers.Dense(
                units=num_units[0],
                activation=activation,
            )
        )
        self._model.add(tf.keras.layers.Dropout(0.25))

        #  Hidden layers
        for hl in range(2, hidden_layers):
            self._model.add(
                tf.keras.layers.Dense(
                    units=num_units[hl - 1],
                    activation=activation,
                )
            )
            self._model.add(tf.keras.layers.Dropout(0.25))

        # Output layer
        self._model.add(
            tf.keras.layers.Dense(units=DeepModel.n_classes, activation="softmax")
        )

        local_cross_entropy = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.2, gamma=2.0)
        cce = tf.keras.losses.CategoricalCrossentropy()
        p = tf.keras.losses.Poisson()
        kl = tf.keras.losses.KLDivergence()

        self._model.compile(
            loss=local_cross_entropy,
            loss_weights=self.get_class_weight().values(),
            optimizer=optimizer,
            metrics=["accuracy"],
        )

    def create_GRU_RNN(
        self,
    ):
        self._model = tf.keras.models.Sequential()
        self._model.add(
            tf.keras.layers.GRU(
                units=128,
                dropout=0.1,
                recurrent_dropout=0.5,
                return_sequences=True,
            )
        )
        self._model.add(tf.keras.layers.Dense(units=DeepModel.n_classes))

        self._model.compile(optimizer=tf.keras.optimizers.RMSprop(), loss="mae")

    def train_NN(self):
        self._history = self._model.fit(
            self._X,
            self._y,
            epochs=DeepModel.epochs,
            batch_size=DeepModel.batch_size,
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
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        # plt.show()

    def get_class_weight(self) -> dict:
        samples = self._y.shape[0]
        unique, counts = np.unique(np.argmax(self._y, axis=1), return_counts=True)
        return dict(zip(unique, 100 - (counts / samples) * 100))

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

        # plt.show()

    def _load_data(self):
        self._X, X_test = np.load("..\\data\\X_train_bert.npy"), np.load(
            "..\\data\\X_test_bert.npy"
        )
        self._y, y_test = np.load("..\\data\\y_train_bert.npy"), np.load(
            "..\\data\\y_test_bert.npy"
        )
        return X_test, y_test

    @staticmethod
    def main() -> None:
        from argsparser import ArgsParser

        X_test, y_test = DeepModel.instance()._load_data()

        args = ArgsParser.get_args()
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

        DeepModel.instance().train_NN()
        DeepModel.instance().evaluate_NN(X=X_test, y=y_test)
        DeepModel.instance().show_history()


if __name__ == "__main__":
    DeepModel.instance().main()

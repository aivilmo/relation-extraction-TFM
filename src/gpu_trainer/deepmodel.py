#!/usr/bin/env python

import numpy as np

# from os import environ

# environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from tensorflow.keras import layers, models, metrics, optimizers


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

        self._model = models.Sequential()

        # Input layer
        self._model.add(
            layers.Dense(
                units=num_units[0],
                activation=activation,
            )
        )
        self._model.add(layers.Dropout(0.25))

        #  Hidden layers
        for hl in range(2, hidden_layers):
            self._model.add(
                layers.Dense(
                    units=num_units[hl - 1],
                    activation=activation,
                )
            )
            self._model.add(layers.Dropout(0.25))

        # Output layer
        self._model.add(layers.Dense(units=DeepModel.n_classes, activation="softmax"))

        local_cross_entropy = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.2, gamma=2.0)
        self._model.compile(
            loss=local_cross_entropy, optimizer=optimizer, metrics=["accuracy"]
        )

    def create_GRU(
        self,
    ):
        self._model = models.Sequential()
        self._model.add(
            layers.GRU(
                units=128,
                dropout=0.1,
                recurrent_dropout=0.5,
                return_sequences=True,
            )
        )
        self._model.add(
            layers.GRU(units=256, activation="relu", dropout=0.1, recurrent_dropout=0.5)
        )
        self._model.add(layers.Dense(units=DeepModel.n_classes))

        self._model.compile(optimizer=optimizers.RMSprop(), loss="mae")

    def train_NN(self, X, y):
        self._history = self._model.fit(
            X,
            y,
            epochs=DeepModel.epochs,
            batch_size=DeepModel.batch_size,
            validation_split=0.2,
            callbacks=DeepModel.get_callbacks(),
            class_weight=DeepModel.get_class_weight(y),
        )

    def evaluate_NN(self, X, y):
        from sklearn.metrics import classification_report

        print("Testing model...")

        y_hat = self._model.predict(X)
        y_hat = np.argmax(y_hat, axis=1)
        y = np.argmax(y, axis=1)

        print("Classification report:")
        print(classification_report(y, y_hat))

    @staticmethod
    def get_class_weight(y: np.ndarray) -> dict:
        samples = y.shape[0]
        unique, counts = np.unique(np.argmax(y, axis=1), return_counts=True)
        return dict(zip(unique, 100 - (counts / samples) * 100))

    @staticmethod
    def get_callbacks() -> list:
        from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

        accuracy = EarlyStopping(
            monitor="val_accuracy", patience=10, restore_best_weights=True
        )
        loss = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)

        # Create a callback that saves the model's weights
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint = ModelCheckpoint(
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

    @staticmethod
    def main() -> None:
        X_train, X_test = np.load("..\\data\\X_train_bert.npy"), np.load(
            "..\\data\\X_test_bert.npy"
        )
        y_train, y_test = np.load("..\\data\\y_train_bert.npy"), np.load(
            "..\\data\\y_test_bert.npy"
        )

        # X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
        # X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])

        DeepModel.instance().create_simple_NN()
        DeepModel.instance().train_NN(X=X_train, y=y_train)
        DeepModel.instance().evaluate_NN(X=X_test, y=y_test)
        DeepModel.instance().show_history()


if __name__ == "__main__":
    DeepModel.instance().main()

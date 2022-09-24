#!/usr/bin/env python

import random
import numpy as np
import tensorflow as tf
from enum import Enum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from model.abstractmodel import AbstractModel, ModelType
from utils.appconstants import AppConstants

seed = 6
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

# from os import environ

# environ["KERAS_BACKEND"] = "plaidml.keras.backend"


class Loss(Enum):
    FOCAL_CROSS_ENTROPY = "sigmoid_focal_crossentropy"
    BINARY_CROSS_ENTROPY = "binary_crossentropy"


class DeepModel(AbstractModel):

    _instance = None

    _epochs = 80
    _batch_size = 64

    @staticmethod
    def instance():
        if DeepModel._instance is None:
            DeepModel()
        return DeepModel._instance

    def __init__(self) -> None:
        super().__init__()
        if DeepModel._instance is not None:
            raise Exception

        self._loss = None
        self._optimizer = None
        self._history = None
        self._n_classes = None
        self._is_multi = False
        self._lda = LinearDiscriminantAnalysis(n_components=11)

        DeepModel._instance = self

    def start_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model,
    ) -> None:
        from utils.preprocess import Preprocessor

        bin_y_train, bin_y_test = self.binarize_labels(y_train, y_test)
        bin_y_train, bin_y_test = Preprocessor.instance().prepare_labels(
            bin_y_train, bin_y_test
        )
        # First train with binary cls
        self._logger.info("First model")
        self.build(X=X_train, y=bin_y_train, model=model)
        self.train()
        self.evaluate(X=X_test, y=bin_y_test)

        idx, ent_X_train, ent_X_test, ent_y_train, ent_y_test = self.take_subsample(
            self._yhat, X_train, X_test, y_train, y_test
        )

        if AppConstants.instance()._lda:
            self._logger.info(f"Applying LinearDiscriminantAnalysis with {self._lda}")
            ent_X_train = self._lda.fit_transform(ent_X_train, ent_y_train)
            self._logger.info(f"LinearDiscriminantAnalysis succesfully appliyed")

        ent_y_train, ent_y_test = Preprocessor.instance().prepare_labels(
            ent_y_train, ent_y_test
        )

        # Second train discriminate cls
        self._logger.info("Second model")
        self.build(X=ent_X_train, y=ent_y_train, model=model)
        self.train()
        self.evaluate(X=ent_X_test, y=ent_y_test)

        y_test[idx] = self._yhat + 1
        self._yhat = y_test
        self.export_results()

    def build(self, X: np.ndarray, y: np.ndarray, model) -> None:
        self._n_classes = y.shape[1]
        super().build(X, y)

        if ModelType(model) is ModelType.DENSE:
            self.build_dense()
        if ModelType(model) is ModelType.MULTIINPUT:
            self._is_multi = True
            self._X = self.handle_multi_input(self._X)
            self.build_multi_dense()
        self.compile()

    def train(self) -> None:
        from time import time

        self._logger.info("Training model...")
        start: float = time()

        self._history = self._model.fit(
            self._X,
            self._y,
            epochs=self._epochs,
            batch_size=self._batch_size,
            validation_split=0.2,
            callbacks=self.get_callbacks(),
            # Para RE
            class_weight=self.compute_class_weight_freq(),
        )

        end: float = time() - start
        self._logger.info(f"Model trained, time: {round(end / 60, 2)} minutes")

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        X = self.handle_multi_input(X)
        if AppConstants.instance()._lda:
            try:
                X = self._lda.transform(X)
            except Exception as e:
                self._logger.warning(e)
                self._logger.warning("Not applying LDA on that dataset")

        yhat = self._model.predict(X)
        yhat = np.argmax(yhat, axis=1)
        y = np.argmax(y, axis=1)

        super().evaluate(yhat, y)

    def build_dense(
        self,
        num_units: list = [768, 384],
        activation: str = "relu",
    ) -> None:
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
        for hl in range(len(num_units) - 1):
            self._model.add(
                tf.keras.layers.Dense(
                    units=num_units[hl + 1],
                    activation=activation,
                )
            )
            self._model.add(tf.keras.layers.Dropout(0.5))

        # Output layer
        self._model.add(
            tf.keras.layers.Dense(units=self._n_classes, activation="softmax")
        )

    @classmethod
    def build_multi_dense(self) -> None:
        # define two sets of inputs
        input_embedding = tf.keras.layers.Input(shape=(768 * 2,))
        input_entities = tf.keras.layers.Input(shape=(8,))

        # the first branch operates on the first input
        embedding = tf.keras.layers.Dense(768, activation="relu")(input_embedding)
        embedding = tf.keras.layers.Dense(384, activation="relu")(embedding)
        embedding = tf.keras.layers.Dropout(0.25)(embedding)
        embedding = tf.keras.models.Model(inputs=input_embedding, outputs=embedding)

        # the second branch opreates on the second input
        entities = tf.keras.layers.Dense(4, activation="relu")(input_entities)
        entities = tf.keras.layers.Dropout(0.25)(entities)
        entities = tf.keras.models.Model(inputs=input_entities, outputs=entities)

        # combine the output of the two branches
        combined = tf.keras.layers.concatenate([embedding.output, entities.output])

        # apply a FC layer and then a regression prediction on the
        # combined outputs
        z = tf.keras.layers.Dropout(0.50)(combined)
        z = tf.keras.layers.Dense(15, activation="softmax")(z)
        # our model will accept the inputs of the two branches and
        # then output a single value
        self._model = tf.keras.models.Model(
            inputs=[embedding.input, entities.input], outputs=z
        )
        print(self._model)

    def compile(self) -> None:
        from main import Main

        loss: str = Main.instance()._args.loss
        if loss == "binary_crossentropy":
            self._loss = tf.keras.losses.BinaryCrossentropy()
        if loss == "sigmoid_focal_crossentropy":
            import tensorflow_addons as tfa

            self._loss = tfa.losses.SigmoidFocalCrossEntropy(alpha=0.20, gamma=2.0)

        self._optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

        self._model.compile(
            loss=self._loss,
            optimizer=self._optimizer,
            metrics=["accuracy"],
        )

        print(self._model.summary())

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

    def handle_multi_input(self, X: np.array) -> list[np.array]:
        if not self._is_multi:
            return X

        embedding = np.hstack((X[:, 0:768], X[:, 768 : 768 * 2]))
        entity = np.hstack(
            (X[:, 768 * 2 : 768 * 2 + 4], X[:, 768 * 2 + 4 : 768 * 2 + 8])
        )
        return [embedding, entity]

    def take_subsample(
        self,
        y_hat: np.ndarray,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
    ) -> tuple[np.ndarray]:
        bin_y_train, bin_y_test = self.binarize_labels(y_train, y_hat)

        indices_train = np.where(bin_y_train > 0)[0]
        out_X_train = np.take(X_train, indices_train, axis=0)
        out_y_train = np.take(y_train, indices_train, axis=0) - 1

        indices_test = np.where(bin_y_test > 0)[0]
        out_X_test = np.take(X_test, indices_test, axis=0)
        out_y_test = np.take(y_test, indices_test, axis=0) - 1

        return indices_test, out_X_train, out_X_test, out_y_train, out_y_test

    def binarize_labels(
        self, y_train: np.ndarray, y_test: np.ndarray
    ) -> tuple[np.ndarray]:
        bin_y_train = y_train.copy()
        bin_y_test = y_test.copy()
        bin_y_train[bin_y_train > 0] = 1
        bin_y_test[bin_y_test > 0] = 1

        return bin_y_train, bin_y_test

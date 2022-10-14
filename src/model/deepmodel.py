#!/usr/bin/env python

import random
import numpy as np
import tensorflow as tf
from enum import Enum
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import pickle
from collections import Counter

from model.abstractmodel import AbstractModel, ModelType
from utils.appconstants import AppConstants
from utils.fileshandler import FilesHandler

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
    _checkpoint_path = "training_1/cp.ckpt"

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
        self._lda = LinearDiscriminantAnalysis(n_components=13)
        self._entity_vc_test = None
        self._prob_yhat = None

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
        import sys

        pr_instance = Preprocessor.instance()
        _, self._entity_vc_test, _, _ = FilesHandler.instance().load_training_data(
            "with_entities"
        )
        ori_y_train, ori_y_test = y_train, y_test
        y_train, y_test = pr_instance.prepare_labels(y_train, y_test)

        self.build(X=X_train, y=y_train, model=model)
        self.train(train=False, number="1")
        self.evaluate(X=X_test, y=y_test)
        first_yhat = self._yhat

        idx, ent_X_train, ent_X_test, ent_y_train, ent_y_test = self.take_subsample(
            self._yhat, X_train, X_test, ori_y_train, ori_y_test
        )
        ent_X_train, ent_y_train = self.apply_oversampling(ent_X_train, ent_y_train)

        if AppConstants.instance()._lda:
            # with open("lda.model", "rb") as f:
            #     self._lda = pickle.load(f)
            # ent_X_train = self._lda.transform(ent_X_train)
            self._logger.info(f"Applying LinearDiscriminantAnalysis with {self._lda}")
            ent_X_train = self._lda.fit_transform(ent_X_train, ent_y_train)
            self._logger.info(f"LinearDiscriminantAnalysis succesfully appliyed")
            with open("lda.model", "wb") as f:
                pickle.dump(self._lda, f)

        ent_y_train, ent_y_test = pr_instance.prepare_labels(ent_y_train, ent_y_test)

        # Second train discriminate cls
        self._logger.info("Second model")
        self.build(X=ent_X_train, y=ent_y_train, model=model)
        self.train(train=True, number="2")

        X = self._lda.transform(ent_X_test)
        yhat = self._model.predict(X)

        self._prob_yhat = self._prob_yhat[idx][:, 1:]
        mean_yhat = (self._prob_yhat * 0.55 + yhat * 0.45) / 2.0
        self._yhat = np.argmax(mean_yhat, axis=1)

        first_yhat[first_yhat > 0] = self._yhat + 1
        self._yhat = self.repuntuate_binary_model(first_yhat)
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

    def train(self, train=True, number: str = "1") -> None:
        from time import time

        self._logger.info("Training model...")
        start: float = time()

        if not train:
            if number == "2":
                self._model.load_weights("training_2/cp.ckpt")
            else:
                self._model.load_weights(self._checkpoint_path)
            return

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

    def evaluate(self, X: np.ndarray, y: np.ndarray, repuntuate: bool = False) -> None:
        if AppConstants.instance()._lda:
            try:
                X = self._lda.transform(X)
            except Exception as e:
                self._logger.warning(e)
                self._logger.warning("Not applying LDA on that dataset")

        yhat = self._model.predict(X)
        self._prob_yhat = yhat

        yhat = np.argmax(yhat, axis=1)
        if repuntuate:
            yhat = self.repuntuate_binary_model(yhat)
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
        checkpoint = tf.keras.callbacks.ModelCheckpoint(
            filepath=DeepModel._checkpoint_path, save_weights_only=True, verbose=1
        )
        return [accuracy, loss]  # , checkpoint]

    def handle_multi_input(self, X: np.array) -> list[np.array]:
        emb_size = 768 * 2
        embedding = np.hstack((X[:, 0:768], X[:, 768:emb_size]))
        entity = np.hstack(
            (X[:, emb_size : emb_size + 1], X[:, emb_size + 1 : emb_size + 2])
        )
        return [embedding, entity]

    def apply_oversampling(self, X_train, y_train) -> tuple[np.array]:
        self._n_classes = 14
        self._X = X_train
        self._y = y_train
        counter = Counter(self._y)
        counter = sorted(counter.items(), key=lambda f: f[1], reverse=True)

        for k, v in counter:
            per = v / len(self._y) * 100
            print("Class=%d, n=%d (%.3f%%)" % (k, v, per))

        self.over_sample_data()

        counter = Counter(self._y)
        counter = sorted(counter.items(), key=lambda f: f[1], reverse=True)
        for k, v in counter:
            per = v / len(self._y) * 100
            print("Class=%d, n=%d (%.3f%%)" % (k, v, per))

        return self._X, self._y

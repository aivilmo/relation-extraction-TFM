#!/usr/bin/env python

import numpy as np
from os import environ

environ["KERAS_BACKEND"] = "plaidml.keras.backend"
from keras.layers import Dense
from keras.models import Sequential


class DeepModel:
    _instance = None
    epochs = 2
    batch_size = 32
    n_classes = 13

    @staticmethod
    def instance():
        if DeepModel._instance == None:
            DeepModel()
        return DeepModel._instance

    def __init__(self) -> None:
        if DeepModel._instance != None:
            raise Exception
        self._model = None
        DeepModel._instance = self

    def test_model(self, X, y) -> None:
        from sklearn.metrics import classification_report

        yhat = self._model.predict(X)
        yhat = np.argmax(yhat, axis=1)
        y = np.argmax(y, axis=1)

        print(classification_report(y, yhat))

    def create_NN(
        self, input_dim, hidden_layers, num_units, activation, optimizer, loss
    ):
        self._model = Sequential()
        # Input layer
        self._model.add(
            Dense(
                units=num_units,
                input_dim=input_dim,
                activation=activation,
            )
        )

        # hidden layers
        for _ in range(hidden_layers):
            # num_units /= 2
            self._model.add(
                Dense(
                    units=num_units,
                    input_dim=input_dim,
                    activation=activation,
                )
            )

        # Output layer
        self._model.add(Dense(units=DeepModel.n_classes, activation="softmax"))
        self._model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])
        return self._model

    def train_NN(self, X, y):
        from keras.callbacks import EarlyStopping

        accuracy = EarlyStopping(
            monitor="val_accuracy", patience=3, restore_best_weights=True
        )
        loss = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        self.create_NN(
            input_dim=X.shape[1],
            hidden_layers=3,
            num_units=128,
            activation="sigmoid",
            optimizer="adam",
            loss="binary_crossentropy",
        )
        self._model.fit(
            X,
            y,
            epochs=DeepModel.epochs,
            batch_size=DeepModel.batch_size,
            validation_split=0.2,
        )

    def evaluate_NN(self, X, y):
        from sklearn.metrics import precision_score, recall_score, f1_score

        yhat = self._model.predict(X)
        yhat = np.argmax(yhat, axis=1)
        y = np.argmax(y, axis=1)

        precision = precision_score(
            y, yhat, average="weighted", labels=[i for i in range(0, 12)]
        )
        recall = recall_score(
            y, yhat, average="weighted", labels=[i for i in range(0, 12)]
        )
        f1score = f1_score(
            y, yhat, average="weighted", labels=[i for i in range(0, 12)]
        )

        print(
            "Precision: %0.3f Recall %0.3f F1 score: %0.3f for %s"
            % (precision, recall, f1score, "Neuronal Network")
        )
        return precision, recall, f1score

    def train_best_NN(self, X, y):
        from sklearn.model_selection import GridSearchCV
        from keras.wrappers.scikit_learn import KerasClassifier

        params = dict(
            input_dim=[X.shape[1]],
            hidden_layers=[1],
            num_units=[256, 128],
            activation=["relu"],
            optimizer=["adam"],
            loss=["binary_crossentropy"],
        )

        base_NN = KerasClassifier(
            build_fn=self.create_NN,
            epochs=DeepModel.epochs,
            batch_size=DeepModel.batch_size,
            verbose=1,
        )
        model = GridSearchCV(estimator=base_NN, param_grid=params, n_jobs=-1)

        # Training
        model.fit(X, y)

        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]

        print("Best: %f using %s" % (model.best_score_, model.best_params_))
        for mean, std, params in sorted(
            zip(means, stds, model.cv_results_["params"]), key=lambda x: -x[0]
        ):
            print(
                "Mean test score: %0.3f (+/-%0.03f) for params: %r"
                % (mean, std * 2, params)
            )
        return model

    @staticmethod
    def main() -> None:

        X_train, X_test = np.load("..\\..\\data\\X_train.npy"), np.load(
            "..\\..\\data\\X_test.npy"
        )
        y_train, y_test = np.load("..\\..\\data\\y_train.npy"), np.load(
            "..\\..\\data\\y_test.npy"
        )

        DeepModel.instance().train_best_NN(X=X_train, y=y_train)


if __name__ == "__main__":
    DeepModel.instance().main()

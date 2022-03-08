#!/usr/bin/env python
from cmath import isnan
import numpy as np
from sklearn.preprocessing import StandardScaler
from logger import Logger


class CoreModel:
    _instance = None

    @staticmethod
    def instance():
        if CoreModel._instance == None:
            CoreModel()
        return CoreModel._instance

    def __init__(self) -> None:
        if CoreModel._instance != None:
            raise Exception

        self._model = None
        self._params = {}
        self._labels = []
        self._X = None
        self._y = None
        self._logger = Logger.instance()
        CoreModel._instance = self

    def set_labels(self, labels: list) -> None:
        self._logger.info(f"Setting labels to model: {labels}")
        self._labels = labels

    def set_model(self, model) -> None:
        self._model = model

    def fit_model(self, X, y) -> None:
        self._logger.info(f"Setting data to model, X: {X.shape}, y: {y.shape}")
        self._X = X
        self._y = y

    def train_model(self) -> None:
        # from sklearn.pipeline import make_pipeline
        # from sklearn import svm
        import time

        self._logger.info("Training model...")
        start = time.time()
        # self._model = make_pipeline(StandardScaler(), svm.SVC())
        self._model.fit(self._X, self._y)
        self._logger.info(f"Model trained, time: {(time.time() - start) / 60} minutes")

    def test_model(self, X, y) -> None:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )
        import matplotlib.pyplot as plt

        self._logger.info("Testing model...")
        y_hat = self._model.predict(X)

        self._logger.info("Classification report:")
        self._logger.info(classification_report(y, y_hat, target_names=self._labels))

        self._logger.info("Confusion matrix:")
        cm = confusion_matrix(y, y_hat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._labels)
        disp.plot()
        # plt.show()

    def train_best_model(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import ShuffleSplit

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        model = GridSearchCV(
            estimator=self._model, param_grid=self._params, cv=cv, n_jobs=6, verbose=1
        )
        self._logger.info("Training best model")
        model.fit(self._X, self._y)

        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in sorted(
            zip(means, stds, model.cv_results_["params"]), key=lambda x: -x[0]
        ):
            if not np.isnan(mean):
                self._logger.info(
                    "Mean test score: %0.3f (+/-%0.03f) for params: %r"
                    % (mean, std * 2, params)
                )
        self._model = model

    def start_train(self, X_train, X_test, y_train, y_test):
        from sklearn.svm import LinearSVC, SVC
        from sklearn.linear_model import Perceptron
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier

        # self._logger.info("Scaling data...")
        # scaler = StandardScaler()
        # X_train = scaler.fit_transform(X_train).reshape(X_train.shape)
        # X_test = scaler.transform(X_test).reshape(X_test.shape)
        # self._logger.info(f"Data scaled, new X_train: {X_train.shape}, new X_test: {X_test.shape} ")

        self._params = {
            "penalty": ["l1", "l2"],
            "loss": ["hingue", "squared_hinge"],
            "dual": [True, False],
            "C": [1, 10, 100, 1000],
            "class_weight": [None],
        }

        self._params = {
            "penalty": ["l2"],
            "loss": ["squared_hinge"],
            "dual": [True],
            "C": [100],
            "class_weight": [None],
        }

        self.set_model(LinearSVC())
        self.fit_model(X_train, y_train)
        self.train_model()
        # self.train_best_model()
        self.test_model(X_test, y_test)

    def get_class_weight(self, y: np.ndarray) -> None:
        from sklearn.utils.class_weight import compute_class_weight

        train_classes = np.unique(y)
        class_weight = compute_class_weight(
            class_weight="balanced", classes=train_classes, y=y
        )
        return dict(zip(train_classes, class_weight))

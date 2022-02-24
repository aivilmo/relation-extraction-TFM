#!/usr/bin/env python
import numpy as np


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
        CoreModel._instance = self

    def set_labels(self, labels: list) -> None:
        print(f"Setting labels to model: {labels}")
        self._labels = labels

    def set_model(self, model) -> None:
        self._model = model

    def fit_model(self, X, y) -> None:
        self._X = X
        self._y = y

    def train_model(self) -> None:
        self._model.fit(self._X, self._y)

    def test_model(self, X, y) -> None:
        from sklearn.metrics import classification_report

        y_hat = self._model.predict(X)
        print(classification_report(y, y_hat))

    def train_best_model(self):
        from sklearn.model_selection import GridSearchCV

        model = GridSearchCV(
            estimator=self._model, param_grid=self._params, n_jobs=-1, verbose=0
        )
        model.fit(self._X, self._y)

        means = model.cv_results_["mean_test_score"]
        stds = model.cv_results_["std_test_score"]
        for mean, std, params in sorted(
            zip(means, stds, model.cv_results_["params"]), key=lambda x: -x[0]
        ):
            print(
                "Mean test score: %0.3f (+/-%0.03f) for params: %r"
                % (mean, std * 2, params)
            )
        print()
        self._model = model

    def start_train(self, X_train, X_test, y_train, y_test):
        from sklearn import svm

        self.set_class_weight(y_train)
        self.set_model(svm.LinearSVC(class_weight=self._params["class_weight"]))
        self.fit_model(X_train, y_train)
        self.train_model()
        self.test_model(X_test, y_test)

    def set_class_weight(self, y: np.ndarray) -> None:
        from sklearn.utils.class_weight import compute_class_weight

        train_classes = np.unique(y)
        class_weight = compute_class_weight(
            class_weight="balanced", classes=train_classes, y=y
        )
        self._params = {"class_weight": dict(zip(train_classes, class_weight))}

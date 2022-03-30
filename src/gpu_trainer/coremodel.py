#!/usr/bin/env python
import numpy as np
from logger.logger import Logger
import time


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
        self._logger.info("Training model...")
        start = time.time()

        self._model.fit(
            self._X, self._y
        )  # , sample_weight=self.get_sample_weight(self._y))

        self._logger.info(f"Model trained, time: {(time.time() - start) / 60} minutes")

    def make_pipeline(self) -> None:
        from sklearn.pipeline import Pipeline
        from sklearn.feature_selection import f_classif, SelectKBest

        self._logger.info("Creating a pipeline...")

        self._model = Pipeline(
            steps=[
                ("feature_selection", SelectKBest(score_func=f_classif, k=100)),
                ("classifier", self._model),
            ],
            verbose=True,
        )

        self._logger.info(
            f"Pipeline cretad with {self._model.named_steps['feature_selection']}"
        )
        start = time.time()

        self._model.fit(self._X, self._y)
        self._logger.info(self._model)

        self._logger.info(
            f"Pipeline trained, time: {(time.time() - start) / 60} minutes"
        )

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
        print(classification_report(y, y_hat))

        self._logger.info("Confusion matrix:")
        cm = confusion_matrix(y, y_hat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        plt.show()

    def train_best_model(self):
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import ShuffleSplit

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        model = GridSearchCV(
            estimator=self._model, param_grid=self._params, cv=cv, n_jobs=8, verbose=1
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
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.neighbors import KNeighborsClassifier

        self.set_model(
            RandomForestClassifier(
                criterion="gini",
                min_samples_split=2,
                n_estimators=100,
                max_depth=None,
                max_features="sqrt",
                n_jobs=10,
                class_weight=self.get_class_weight(y_train),
            )
        )
        self.fit_model(X_train, y_train)
        self.make_pipeline()
        # self.train_model()
        # self.train_best_model()
        self.test_model(X_test, y_test)

    def get_class_weight(self, y: np.ndarray) -> dict:
        from sklearn.utils.class_weight import compute_class_weight

        train_classes = np.unique(y)
        class_weight = compute_class_weight(
            class_weight="balanced", classes=train_classes, y=y
        )
        return dict(zip(train_classes, class_weight))

    def get_sample_weight(self, y: np.ndarray) -> dict:
        from sklearn.utils.class_weight import compute_sample_weight

        return compute_sample_weight(class_weight="balanced", y=y)

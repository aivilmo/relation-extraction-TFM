#!/usr/bin/env python

from time import time
import numpy as np
from enum import Enum
from collections import Counter
from matplotlib import pyplot


from logger.logger import Logger
from utils.appconstants import AppConstants


class ModelType(Enum):
    SVM = "svm"
    PRECEPTRON = "perceptron"
    DECISIONTREE = "decisiontree"
    RANDOMFOREST = "randomforest"
    DENSE = "dense"
    MULTIINPUT = "multi"


class AbstractModel:

    _instance = None
    _n_classes = None
    _random_state = 42

    _logger = Logger.instance()
    _task: str = AppConstants.instance()._task
    _imbalance_strategy: str = AppConstants.instance()._imbalance_strategy

    _sampling_strategy: dict = {
        "scenario2-taskA": {7: 100, 4: 350, 6: 250},
        "scenario3-taskB": {7: 100, 4: 350, 6: 250},
    }

    def __init__(self) -> None:
        if AbstractModel._instance is not None:
            raise Exception

        self._model = None
        self._labels: list = []

        self._X: np.ndarray = None
        self._y: np.ndarray = None
        self._yhat: np.ndarray = None

        AbstractModel._instance = self

    def start_training(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        model,
    ) -> None:
        self.build(X=X_train, y=y_train, model=model)
        counter = Counter(self._y)
        for k, v in counter.items():
            per = v / len(self._y) * 100
            print("Class=%d, n=%d (%.3f%%)" % (k, v, per))

        if self._imbalance_strategy == "oversampling":
            self.over_sample_data()
        if self._imbalance_strategy == "undersampling":
            self.under_sample_data()
        if self._imbalance_strategy == "both":
            self.combined_resample_data()

        print()

        counter = Counter(self._y)
        for k, v in counter.items():
            per = v / len(self._y) * 100
            print("Class=%d, n=%d (%.3f%%)" % (k, v, per))

        self.train()
        self.evaluate(X=X_test, y=y_test)
        self.export_results()

    def build(self, X: np.ndarray, y: np.ndarray) -> None:
        AbstractModel._logger.info(f"Building model {self._model}...")
        self._X = X
        self._y = y
        # self._n_classes = len(np.unique(self._y))
        AbstractModel._logger.info(f"Model has built successfully")

    def train(self, **kwargs) -> None:
        self._logger.info("Training model...")
        start: float = time()

        self._model.fit(X=self._X, y=self._y)

        end: float = time() - start
        self._logger.info(f"Model trained, time: {round(end / 60, 2)} minutes")

    def evaluate(self, yhat: np.ndarray, y: np.ndarray) -> None:
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            ConfusionMatrixDisplay,
        )
        import matplotlib.pyplot as plt

        self._logger.info(f"Testing model {self._model}")
        self._yhat = yhat

        self._logger.info("Classification report:")
        print(classification_report(y, self._yhat))

        self._logger.info("Confusion matrix:")
        cm = confusion_matrix(y, self._yhat)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot()
        # plt.show()

    def export_results(self) -> None:
        from sklearn.preprocessing import LabelEncoder
        from utils.fileshandler import FilesHandler
        from main import Main

        self._logger.info("Exporting results to dataframe...")
        fh_instance = FilesHandler.instance()

        train, test = fh_instance.load_datasets()
        y_column = Main.instance().get_y_column()

        le = LabelEncoder()
        le.fit(train[y_column].values)

        try:
            test["predicted_tag"] = le.inverse_transform(self._yhat)
            fh_instance.save_datasets(train, test)
        except Exception as e:
            self._logger.error("Error exporting results")
            self._logger.error(e)

    def compute_sample_weight(self) -> dict:
        from sklearn.utils.class_weight import compute_sample_weight

        return compute_sample_weight(class_weight="balanced", y=self._y)

    def compute_class_weight(self) -> dict:
        from sklearn.utils.class_weight import compute_class_weight

        train_classes: np.array = np.unique(self._y)
        class_weight: np.array = compute_class_weight(
            class_weight="balanced", classes=train_classes, y=self._y
        )
        return dict(zip(train_classes, class_weight))

    # SOURCE:
    # https://towardsdatascience.com/handling-class-imbalanced-data-using-a-loss-specifically-made-for-it-6e58fd65ffab

    def compute_class_weight_imbalanced(self, beta: float = 0.9) -> np.ndarray:
        unique, samples_per_cls = np.unique(
            np.argmax(self._y, axis=1), return_counts=True
        )
        effective_num = 1.0 - np.power(beta, samples_per_cls)
        weights = (1.0 - beta) / np.array(effective_num)
        weights = weights / np.sum(weights) * AbstractModel._n_classes

        return dict(zip(unique, weights))

    def compute_class_weight_freq(self) -> dict:
        samples = self._y.shape[0]
        unique, counts = np.unique(np.argmax(self._y, axis=1), return_counts=True)
        return dict(zip(unique, 100 - (counts / samples) * 100))

    def under_sample_data(self) -> None:
        from imblearn.under_sampling import NearMiss
        from imblearn.over_sampling import SMOTENC

        under_sampler = NearMiss(sampling_strategy={8: 10000})

        self._logger.info(f"Undersampling data with {under_sampler}")
        self._X, self._y = under_sampler.fit_resample(self._X, self._y)

    def over_sample_data(self) -> None:
        from imblearn.over_sampling import SMOTE, SMOTENC

        over_sampler = SMOTENC(
            categorical_features=[True] * self._n_classes,
            sampling_strategy=self._sampling_strategy[self._task],
            n_jobs=10,
        )

        self._logger.info(f"Oversampling data with {over_sampler}")
        self._X, self._y = over_sampler.fit_resample(self._X, self._y)

    def combined_resample_data(self) -> None:
        from imblearn.combine import SMOTETomek, SMOTEENN

        resampler = SMOTETomek(
            sampling_strategy=self._sampling_strategy[self._task],
            random_state=AbstractModel._random_state,
            n_jobs=10,
        )

        self._logger.info(f"Combined resampling data with {resampler}")
        self._X, self._y = resampler.fit_resample(self._X, self._y)

    def set_labels(self, labels: list) -> None:
        self._logger.info(f"Setting labels to model: {labels}")
        self._labels = labels

#!/usr/bin/env python
import numpy as np
import time
from model.abstractmodel import AbstractModel, ModelType
from sklearn import svm, linear_model, tree, ensemble


class CoreModel(AbstractModel):

    _instance = None

    _available_models: dict = {
        ModelType.SVM: svm.LinearSVC(class_weight="balanced"),
        ModelType.PRECEPTRON: linear_model.Perceptron(class_weight="balanced"),
        ModelType.DECISIONTREE: tree.DecisionTreeClassifier(),
        ModelType.RANDOMFOREST: ensemble.RandomForestClassifier(
            class_weight="balanced"
        ),
    }

    @staticmethod
    def instance():
        if CoreModel._instance is None:
            CoreModel()
        return CoreModel._instance

    def __init__(self) -> None:
        super().__init__()
        if CoreModel._instance is not None:
            raise Exception

        CoreModel._instance = self

    # def start_training(
    #     self,
    #     X_train: np.ndarray,
    #     X_test: np.ndarray,
    #     y_train: np.ndarray,
    #     y_test: np.ndarray,
    #     model,
    # ) -> None:
    #     bin_y_train, bin_y_test = self.binarize_labels(y_train, y_test)

    #     # First train with binary cls
    #     self._logger.info("First model")
    #     self.build(X=X_train, y=bin_y_train, model=model)
    #     self.train()

    #     """
    #     2          2          2      143222
    #     1          2          2       85763
    #     2          1          2       85763
    #     1          1          2       41866
    #     0          2          2       30630
    #     2          0          2       30630
    #     """

    #     yhat = self._model.predict(X_test)
    #     yhat = self.repuntuate_binary_model(X_test, yhat)
    #     super().evaluate(yhat, bin_y_test)

    def build(self, X: np.ndarray, y: np.ndarray, model, **kwargs) -> None:
        super().build(X, y)
        self._model = self._available_models[ModelType(model)]

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> None:
        yhat = self._model.predict(X)

        super().evaluate(yhat, y)

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
            f"Pipeline created with {self._model.named_steps['feature_selection']}"
        )
        start: float = time.time()

        self._model.fit(self._X, self._y)
        self._logger.info(self._model)

        self._logger.info(
            f"Pipeline trained, time: {(time.time() - start) / 60} minutes"
        )

    def train_best_model(self) -> None:
        from sklearn.model_selection import GridSearchCV
        from sklearn.model_selection import ShuffleSplit

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)
        self._model = GridSearchCV(
            estimator=self._model, param_grid=self._params, cv=cv, n_jobs=8, verbose=1
        )
        self._logger.info("Training best model")
        self._model.fit(self._X, self._y)

        means = self._model.cv_results_["mean_test_score"]
        stds = self._model.cv_results_["std_test_score"]
        for mean, std, params in sorted(
            zip(means, stds, self._model.cv_results_["params"]), key=lambda x: -x[0]
        ):
            if not np.isnan(mean):
                self._logger.info(
                    "Mean test score: %0.3f (+/-%0.03f) for params: %r"
                    % (mean, std * 2, params)
                )

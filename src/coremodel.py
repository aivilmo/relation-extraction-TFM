#!/usr/bin/env python


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
        self._params = None
        self._X = None
        self._y = None
        CoreModel._instance = self

    def set_params(self, params) -> None:
        self._params = params

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
            estimator=self._model,
            param_grid=self._params,
            n_jobs=-1,
            verbose=0,
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

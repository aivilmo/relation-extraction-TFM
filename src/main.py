#!/usr/bin/env python
from fileshandler import FilesHandler
import pandas as pd
from pathlib import Path
import argparse


class Main:

    text_path: str = "..\\dataset\\ehealthkd_CONCEPT_ACT_PRED_relaciones\\2021\\ref"
    output_train: str = "data\\train.pkl"
    output_test: str = "data\\test.pkl"

    @staticmethod
    def main() -> None:
        args = Main.__get_args()

        if args.generate:
            dataset_train: pd.DataFrame = FilesHandler.generate_dataset(
                Path(Main.text_path + "\\training\\"), Main.output_train
            )
            dataset_test: pd.DataFrame = FilesHandler.generate_dataset(
                Path(Main.text_path + "\\testing\\"), Main.output_test
            )
        elif args.load:
            dataset_train: pd.DataFrame = FilesHandler.load_dataset(Main.output_train)
            dataset_test: pd.DataFrame = FilesHandler.load_dataset(Main.output_test)

        if args.visualization:
            from visualizationhandler import VisualizationHandler

            VisualizationHandler.visualice_relations(dataset_train)
            VisualizationHandler.visualice_most_common_words(dataset_train, n_words=10)
        elif args.train:
            from coremodel import CoreModel
            from preprocess import Preprocessor
            from sklearn.linear_model import Perceptron

            prep = Preprocessor()
            X_train, X_test, y_train, y_test = prep.train_test_split(
                dataset_train, dataset_test
            )

            core = CoreModel()
            core.set_model(Perceptron())
            core.fit_model(X_train, y_train)
            core.train_model()
            core.test_model(X_test, y_test)

    @staticmethod
    def __get_args():
        parser = argparse.ArgumentParser()

        data = parser.add_mutually_exclusive_group(required=True)
        data.add_argument(
            "--generate",
            action="store_true",
            help="If you want to generate a dataset",
        )

        data.add_argument(
            "--load",
            action="store_true",
            help="If you want to load a generated dataset",
        )

        action = parser.add_mutually_exclusive_group()
        action.add_argument(
            "--visualization",
            action="store_true",
            help="If you want to visualice the dataset",
        )

        action.add_argument(
            "--train",
            action="store_true",
            help="If you want to train a model",
        )

        return parser.parse_args()


if __name__ == "__main__":
    Main.main()

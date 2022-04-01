#!/usr/bin/env python

import argparse


class ArgsParser:
    @staticmethod
    def get_args() -> argparse.Namespace:
        parser: argparse.ArgumentParser = argparse.ArgumentParser()

        # Generate or load a dataset
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

        # Visualizes data information or train a model
        action = parser.add_mutually_exclusive_group()
        action.add_argument(
            "--visualization",
            action="store_true",
            help="If you want to visualize the dataset",
        )

        action.add_argument(
            "--train",
            action="store_true",
            help="If you want to train a model",
        )

        # Select kind of model
        model = parser.add_mutually_exclusive_group()
        model.add_argument(
            "--ml_model",
            nargs=1,
            action="store",
            choices=[
                "svm",
                "perceptron",
                "decisiontree",
                "randomforest",
            ],
            help="Select ML model to train",
        )

        model.add_argument(
            "--dl_model",
            nargs=1,
            action="store",
            choices=[
                "dense",
                "gru",
            ],
            help="Select DL model to train",
        )

        # Add a list of features to train the model
        parser.add_argument(
            "--features",
            nargs="+",
            action="store",
            default="PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
            choices=[
                "with_entities",
                "sent_emb",
                "word_emb",
                "bag_of_words",
                "single_word_emb",
                "tf_idf",
                "chars",
                "tokens",
                "distilbert-base-uncased",
                "distilbert-base-cased",
                "bert-base-uncased",
                "bert-base-cased",
                "bert-base-multilingual-uncased",
                "bert-base-multilingual-cased",
                "dccuchile/bert-base-spanish-wwm-uncased",
                "dccuchile/bert-base-spanish-wwm-cased",
                "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                "ixa-ehu/ixambert-base-cased",
                "gpt2",
            ],
            help="If you want to add custom features to train",
        )

        # Select a loss function
        parser.add_argument(
            "--loss",
            nargs=1,
            action="store",
            default="sigmoid_focal_crossentropy",
            choices=["sigmoid_focal_crossentropy", "binary_crossentropy"],
            help="Select the loss function to load",
        )

        # Select a imbalance strategy
        parser.add_argument(
            "--imbalance_strategy",
            nargs=1,
            action="store",
            default=None,
            choices=["oversampling", "undersampling", "both"],
            help="Select the sampling strategy for fight against imbalance of data.",
        )

        return parser.parse_args()

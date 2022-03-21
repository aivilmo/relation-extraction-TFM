#!/usr/bin/env python

import argparse


class ArgsParser:
    @staticmethod
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        # Select kind of model to train
        parser.add_argument(
            "--model",
            nargs=1,
            action="store",
            choices=[
                "basic_nn",
                "gru",
                "lstm",
            ],
            help="Select model to train",
        )

        parser.add_argument(
            "--features",
            nargs=1,
            action="store",
            choices=[
                "with_entities",
                "sent_emb",
                "word_emb",
                "bag_of_words",
                "single_word_emb",
                "tf_idf",
                "chars",
                "distilbert-base-uncased",
                "distilbert-base-cased",
                "bert-base-uncased",
                "bert-base-cased",
                "bert-base-multilingual-uncased",
                "bert-base-multilingual-cased",
                "dccuchile/bert-base-spanish-wwm-uncased",
                "dccuchile/bert-base-spanish-wwm-cased",
                "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                "gpt2",
            ],
            help="Selecte the feature to load",
        )

        parser.add_argument(
            "--loss",
            nargs=1,
            action="store",
            choices=["sigmoid_focal_crossentropy", "binary_crossentropy"],
            help="Selecte the loss function to load",
        )

        parser.add_argument(
            "--imbalance_strategy",
            nargs=1,
            action="store",
            choices=["oversampling", "undersampling", "both"],
            help="Selecte the sampling strategy for fight against imabalnce of data.",
        )

        return parser.parse_args()

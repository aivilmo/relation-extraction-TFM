#!/usr/bin/env python

import argparse


class ArgsParser:
    @staticmethod
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

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

        # Visualices data information or train a model
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

        # Add a list of features to train the model
        parser.add_argument(
            "--features",
            nargs="+",
            action="store",
            choices=[
                "with_entities",
                "word_dist",
                "sent_emb",
                "word_emb",
                "bag_of_words",
            ],
            help="If you want to add custom features to train",
        )

        return parser.parse_args()

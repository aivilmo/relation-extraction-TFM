#!/usr/bin/env python

import argparse


class ArgsParser:
    @staticmethod
    def get_args() -> argparse.Namespace:
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

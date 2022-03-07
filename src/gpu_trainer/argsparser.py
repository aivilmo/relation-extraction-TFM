#!/usr/bin/env python

import argparse


class ArgsParser:
    @staticmethod
    def get_args() -> argparse.Namespace:
        parser = argparse.ArgumentParser()

        # Select kind of model to train
        parser.add_argument(
            "--model",
            nargs="+",
            action="store",
            choices=[
                "basic_nn",
                "gru",
                "lstm",
            ],
            help="Select model to train",
        )

        return parser.parse_args()

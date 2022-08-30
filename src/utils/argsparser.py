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
            action="store",
            choices=[
                "dense",
                "gru",
            ],
            help="Select DL model to train",
        )

        model.add_argument(
            "--s2s_model",
            action="store_true",
            help="Select Sequence to Sequence model to train",
        )

        model.add_argument(
            "--transformer_model",
            action="store_true",
            help="Select Transformer model to train",
        )

        # Add a list of features to train the model
        parser.add_argument(
            "--features",
            nargs="+",
            action="store",
            default=["PlanTL-GOB-ES/roberta-base-biomedical-clinical-es"],
            choices=[
                "with_entities",
                "sent_emb",
                "word_emb",
                "bag_of_words",
                "single_word_emb",
                "tf_idf",
                "chars",
                "tokens",
                "seq2seq",
                "bert-base-multilingual-cased",
                "dccuchile/bert-base-spanish-wwm-cased",
                "PlanTL-GOB-ES/roberta-base-biomedical-es",
                "PlanTL-GOB-ES/roberta-base-biomedical-clinical-es",
                "PlanTL-GOB-ES/bsc-bio-ehr-es-pharmaconer",
                "data\\scenario2-taskA\\models\\40epoch\\bert-base-multilingual-cased",
                "pos_tag",
            ],
            help="If you want to add custom features to train",
        )

        # Select a loss function
        parser.add_argument(
            "--loss",
            action="store",
            default="binary_crossentropy",
            choices=["sigmoid_focal_crossentropy", "binary_crossentropy"],
            help="Select the loss function to load",
        )

        # Select a imbalance strategy
        parser.add_argument(
            "--imbalance_strategy",
            action="store",
            default=None,
            choices=["oversampling", "undersampling", "both"],
            help="Select the sampling strategy for fight against imbalance of data.",
        )

        # Select NER, RE or both
        parser.add_argument(
            "--task",
            action="store",
            default="scenario3-taskB",
            choices=["scenario2-taskA", "scenario3-taskB"],
            help="Select the task you want to do",
        )

        # Select directory on save the output file
        parser.add_argument(
            "--run",
            action="store",
            default="1",
            choices=["1", "2", "3"],
            help="Select the directory number to export the ann file results",
        )

        # Export results to ann file
        parser.add_argument(
            "--export",
            action="store_true",
            help="Select if you want to export results to ann file",
        )

        parser.add_argument(
            "--test_dataset",
            action="store",
            default="testing",
            choices=["training", "develop", "testing"],
            help="Select the test dataset for the model",
        )

        parser.add_argument(
            "--data_aug",
            action="store",
            choices=["back_translation", "synonym"],
            help="Select the data augmentation strategy",
        )

        return parser.parse_args()

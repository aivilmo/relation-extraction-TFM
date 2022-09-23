#!/usr/bin/env python
import argparse

from logger.logger import Logger


class AppConstants:

    _instance = None

    _logger = Logger.instance()

    @staticmethod
    def instance():
        if AppConstants._instance is None:
            AppConstants()
        return AppConstants._instance

    def __init__(self) -> None:
        from utils.argsparser import ArgsParser

        if AppConstants._instance is not None:
            raise Exception

        self._args: argparse.Namespace = ArgsParser.get_args()
        self._features = self._args.features
        self._task = self._args.task
        self._run = self._args.run
        self._test_dataset = self._args.test_dataset
        self._data_aug = self._args.data_aug
        self._imbalance_strategy = self._args.imbalance_strategy
        self._lda = self._args.lda

        self._logger.info("Application args parsed")
        AppConstants._instance = self

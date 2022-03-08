import logging
import sys


class Logger:

    _instance = None

    @staticmethod
    def instance():
        if Logger._instance == None:
            Logger()
        return Logger._instance

    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[
                logging.FileHandler("logger\\core_app.log"),
                logging.StreamHandler(),
            ],
        )

        self._logger = logging.getLogger()
        Logger._instance = self

    def debug(self, message):
        # print("[DEBUG]: " + message)
        self._logger.debug(message)

    def info(self, message):
        # print("[INFO]: " + message)
        self._logger.info(message)

    def warning(self, message):
        # print("[WARNING]: " + message)
        self._logger.warning(message)

    def error(self, message):
        # print("[ERROR]: " + message)
        self._logger.error(message)

    def critical(self, message):
        # print("[CRITICAL]: " + message)
        self._logger.critical(message)

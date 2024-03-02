import logging
import os
import sys

logger = logging.getLogger(__name__)


def main():

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    pass


if __name__ == "__main__":
    main()

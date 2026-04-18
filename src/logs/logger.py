import logging
import os

def setup_logger(name="Agentic-RAG"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s"
    )

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)

    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(log_dir, "Agentic-RAG.log")

    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    logger.propagate = False

    return logger

logger = setup_logger()

import logging
import os

def setup_logger(name="Agentic-RAG"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    # Clear existing handlers if any to avoid duplication
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s"
    )

    ch = logging.StreamHandler()
    # handlers just deliver the log where ? stream -> in CMD
    ch.setFormatter(formatter)

    # Get the logs directory relative to this script
    log_dir = os.path.dirname(os.path.abspath(__file__))
    log_file_path = os.path.join(log_dir, "Agentic-RAG.log")

    fh = logging.FileHandler(log_file_path, encoding='utf-8')
    # deliver the logs in file 
    fh.setFormatter(formatter)

    logger.addHandler(ch)
    logger.addHandler(fh)

    # Disable propagation to avoid double logging 
    logger.propagate = False

    return logger

logger = setup_logger()

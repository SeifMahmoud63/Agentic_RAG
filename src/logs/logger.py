import logging

def setup_logger(name="Agentic-RAG"):

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] - %(message)s"
    )


    ch = logging.StreamHandler()
    # handlers just deliver the log where ? stream -> in CMD
    ch.setFormatter(formatter)

    fh = logging.FileHandler("Agentic-RAG.log")
    # deliver the logs in file 
    fh.setFormatter(formatter)

    if not logger.hasHandlers():
        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger



logger=setup_logger()
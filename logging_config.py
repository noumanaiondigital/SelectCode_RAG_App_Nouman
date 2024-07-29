import logging


def setup_logging(log_file='app.log', level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # Add this line to also display logs on the console
        ]
    )

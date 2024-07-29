import logging
from logging_config import setup_logging  # Import your logging configuration

# Set up logging
setup_logging()

logger = logging.getLogger(__name__)


def get_md_content(documents):
    """
    Extract content from a list of Markdown documents.

    Args:
        documents (list): A list of Markdown document files.

    Returns:
        str: The concatenated content of all the documents.
    """
    raw_text = ""
    try:
        for document in documents:
            content = document.read().decode('utf-8')
            raw_text += content + "\n"
        logger.info("Markdown documents processed successfully.")
    except Exception as e:
        logger.error(f"Error processing Markdown documents: {e}")
        raise

    return raw_text

import streamlit as st
from web_template import css, user_template, bot_template
from rag_utils import get_chunks, get_embeddings_and_retrievers
from preprocess_files import get_md_content
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI
from langchain.retrievers import EnsembleRetriever
from rag_test import run_rag_test
import os
import configparser
import warnings
import logging
from logging_config import setup_logging  # Import your logging configuration

warnings.filterwarnings("ignore")

# Set up logging
setup_logging()

logger = logging.getLogger(__name__)

# Load configuration settings from env.cfg file
cfg = configparser.ConfigParser()
try:
    cfg.read('env.cfg')
    openai_key = cfg.get('KEYS', 'API_KEY', raw='')
    flag_run_rag_tests = cfg.get('OTHERS', 'RUN_RAG_TEST', raw='')
    logger.info("Configuration settings loaded successfully.")
except Exception as e:
    logger.error(f"Error loading configuration settings: {e}")
    raise

# Set environment variables for OpenAI API key and timeout settings
try:
    os.environ["OPENAI_API_KEY"] = openai_key
    os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "30"
    logger.info("Environment variables set successfully.")
except Exception as e:
    logger.error(f"Error setting environment variables: {e}")
    raise


def start_conversation(ensemble_retriever: EnsembleRetriever) -> ConversationalRetrievalChain:
    """
    Start a conversational retrieval chain using the given ensemble retriever.

    Args:
        ensemble_retriever (EnsembleRetriever): The ensemble retriever to be used in the conversation.

    Returns:
        ConversationalRetrievalChain: The initialized conversational retrieval chain.
    """
    try:
        # Initialize the language model
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
            request_timeout=20
        )

        # Initialize the conversation memory buffer
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
        )

        # Create the conversational retrieval chain
        conversation = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=ensemble_retriever,
            memory=memory
        )

        logger.info("Conversational retrieval chain initialized successfully.")
        return conversation
    except Exception as e:
        logger.error(f"Error initializing conversational retrieval chain: {e}")
        raise


def process_query(query_text: str) -> None:
    """
    Process the user's query and update the chat history.

    Args:
        query_text (str): The user's query text.

    Returns:
        None
    """
    try:
        response = st.session_state.conversation({'question': query_text})
        st.session_state.chat_history = response["chat_history"]

        for i, message in enumerate(st.session_state.chat_history):
            if i % 2 == 0:
                st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            else:
                st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        logger.info("Query processed successfully.")
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        st.error("An error occurred while processing your query. Please try again.")


def main() -> None:
    """
    Main function to run the Streamlit app.

    Returns:
        None
    """
    try:
        st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
        st.write(css, unsafe_allow_html=True)
        st.header("Hi, I am SelectCode's ChatBot")

        query = st.text_input("How can I help you today?")
        if query:
            process_query(query)

        if "conversation" not in st.session_state:
            st.session_state.conversation = None

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        with st.sidebar:
            st.subheader("Markdown documents")
            documents = st.file_uploader(
                "Upload your Markdown files", type=["md"], accept_multiple_files=True
            )
            if st.button("Run"):
                with st.spinner("Processing..."):
                    try:
                        # Extract text from Markdown documents
                        extracted_text = get_md_content(documents)
                        # Convert text to chunks of data
                        text_chunks = get_chunks(extracted_text)
                        # Create vector embeddings and ensemble retriever
                        ensemble_retriever = get_embeddings_and_retrievers(text_chunks)
                        # Create conversation
                        st.session_state.conversation = start_conversation(ensemble_retriever)
                        logger.info("Markdown documents processed successfully.")
                    except Exception as e:
                        logger.error(f"Error processing Markdown documents: {e}")
                        st.error("An error occurred while processing the documents. Please try again.")
    except Exception as e:
        logger.error(f"Error in main function: {e}")
        st.error("An error occurred in the application. Please try again later.")


if __name__ == "__main__":
    main()

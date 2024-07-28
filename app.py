import streamlit as st
from web_template import css, user_template, bot_template
from rag_utils import get_chunks, get_embeddings_and_retrievers
from preprocess_files import get_md_content
from llm_helpers import start_conversation
from rag_test import run_rag_test
import os

import configparser
cfg = configparser.ConfigParser()
import warnings
warnings.filterwarnings("ignore")


cfg.read('env.cfg')
openai_key = cfg.get('KEYS', 'API_KEY', raw='')
flag_run_rag_tests = cfg.get('OTHERS', 'RUN_RAG_TEST', raw='')

os.environ["OPENAI_API_KEY"] = openai_key
os.environ["PYDEVD_WARN_EVALUATION_TIMEOUT"] = "30"




def process_query(query_text):
    response = st.session_state.conversation({'question': query_text})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)


def main():
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
                # extract text from pdf documents
                extracted_text = get_md_content(documents)
                # convert text to chunks of data
                text_chunks = get_chunks(extracted_text)
                # create vector embeddings
                # vector_embeddings = get_embeddings(text_chunks)

                ensemble_retriever = get_embeddings_and_retrievers(text_chunks)

                # create conversation
                st.session_state.conversation = start_conversation(ensemble_retriever)

if __name__ == "__main__":
    main()


# print("Testing started")
# if flag_run_rag_tests=="True":
#     file_path = cfg.get('OTHERS', 'RAG_FILE_PATH', raw='')
#     eval_questions_path = cfg.get('OTHERS', 'QUESTIONS_FILE_PATH', raw='')
#     run_rag_test(file_path, eval_questions_path)
#     print("Testing completed, files saved.")
import streamlit as st
from langchain.text_splitter import CharacterTextSplitter


from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

from web_template import css, user_template, bot_template

import os

import configparser
cfg = configparser.ConfigParser()
cfg.read('env.cfg')
openai_key = cfg.get('KEYS', 'API_KEY', raw='')

os.environ["OPENAI_API_KEY"] = openai_key

def get_md_content(documents):
    raw_text = ""

    for document in documents:
        content = document.read().decode('utf-8')
        raw_text += content + "\n"

    return raw_text


def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks



def get_embeddings(chunks):
    embeddings = OpenAIEmbeddings()
    # embeddings = HuggingFaceInstructEmbeddings(model_name="")
    vector_storage = FAISS.from_texts(texts=chunks, embedding=embeddings)

    return vector_storage


def start_conversation(vector_embeddings):
    llm = ChatOpenAI()
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_embeddings.as_retriever(),
        memory=memory
    )

    return conversation

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
                vector_embeddings = get_embeddings(text_chunks)
                # create conversation
                st.session_state.conversation = start_conversation(vector_embeddings)



# if __name__ == "__main__":
#     main()
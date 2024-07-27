import streamlit as st
from web_template import css

def main():
    st.set_page_config(page_title="Chat with PDFs", page_icon=":books:", layout="wide")
    st.write(css, unsafe_allow_html=True)
    st.header("Hi, I am SelectCode's ChatBot")
    query = st.text_input("How can I help you today?")
    if query:
        pass




    with st.sidebar:
        st.subheader("Markdown documents")
        documents = st.file_uploader(
            "Upload your Markdown files", type=["md"], accept_multiple_files=True
        )
        


    



if __name__ == "__main__":
    main()
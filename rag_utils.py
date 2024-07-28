from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from typing import List

def get_chunks(text: str) -> List[str]:
    """
    Split the given text into chunks using a character-based text splitter.

    Args:
        text (str): The text to be split into chunks.

    Returns:
        List[str]: A list of text chunks.
    """
    text_splitter = CharacterTextSplitter(
        separator="##",
        chunk_size=4000,
        chunk_overlap=2000,
        length_function=len
    )
    text_chunks = text_splitter.split_text(text)
    return text_chunks

def get_embeddings_and_retrievers(chunks: List[str]) -> EnsembleRetriever:
    """
    Generate embeddings for text chunks and create an ensemble retriever using BM25 and FAISS retrievers.

    Args:
        chunks (List[str]): The text chunks for which embeddings and retrievers are to be created.

    Returns:
        EnsembleRetriever: An ensemble retriever combining BM25 and FAISS retrievers.
    """
    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Initialize FAISS vector store
    faiss_vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    faiss_retriever = faiss_vectorstore.as_retriever(search_kwargs={"k": 5})

    # Initialize BM25 retriever
    bm25_retriever = BM25Retriever.from_texts(texts=chunks, k=5)

    # Initialize ensemble retriever
    ensemble_retriever = EnsembleRetriever(retrievers=[bm25_retriever, faiss_retriever], weights=[0.5, 0.5])

    return ensemble_retriever

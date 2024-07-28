from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOpenAI

def start_conversation(ensemble_retriever):
    llm = ChatOpenAI(model="gpt-4o-mini",
                     temperature=0.2,
                     request_timeout=20)
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=ensemble_retriever,
        memory=memory
    )
    return conversation
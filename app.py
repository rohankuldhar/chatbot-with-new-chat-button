# tempelate for chat with chat history and new chat button

import streamlit as st
import pickle
from streamlit_extras.add_vertical_space import add_vertical_space
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts.prompt import PromptTemplate
from dotenv import load_dotenv
import os
st.write(
	"Has environment variables been set:",
	os.environ["OPENAI_API_KEY"] == st.secrets["OPENAI_API_KEY"])

# Define a variable to store the chat history in the Streamlit session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# "New Chat" button
if st.button("New Chat", key="start_new_chat", help="Click to start a new chat"):
    st.session_state.is_chatting = True
    st.session_state.messages = []

# Storing history in session states
if "messages" not in st.session_state:
    st.session_state.messages = []
for message in st.session_state["messages"]:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    elif message["role"] == "assistant":
        with st.chat_message("assistant"):
            st.markdown(message["content"])

# upload a PDF file
pdf = st.file_uploader("Upload your PDF", type='pdf')

def main(pdf):
    load_dotenv()

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)

        # # embeddings
        store_name = pdf.name[:-4]

        
        embeddings = OpenAIEmbeddings()
        VectorStore = FAISS.from_texts(chunks, embedding=embeddings)

        query = st.chat_input(placeholder="Ask questions about your PDF file:")

        # Check if the "New Chat" button was clicked to reset the chat state
        if st.session_state.start_new_chat:
            st.session_state.is_chatting = True
            st.session_state.messages = []  # Reset chat history

        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        if query:
            chat_history = []
            with st.chat_message("user"):
                st.markdown(query)
            st.session_state.messages.append({"role": "user", "content": query})

            llm = OpenAI(temperature=0)

            qa = ConversationalRetrievalChain.from_llm(
                llm,
                VectorStore.as_retriever(),
                memory=memory
            )
            response = qa({"question": query, "chat_history": chat_history})

            with st.chat_message("assistant"):
                st.markdown(response["answer"])
            st.session_state.messages.append({"role": "assistant", "content": response["answer"]})
            st.session_state.chat_history.append((query, response['answer']))


# In the sidebar, display the chat history
with st.sidebar:
    st.title('Chat History')
    for i, (user_msg, bot_response) in enumerate(st.session_state.chat_history):
        with st.expander(f"Chat {i + 1}"):
            st.markdown(f"User: {user_msg}")
            st.markdown(f"Assistant: {bot_response}")
            

if __name__ == '__main__':
    main(pdf)

#UI for Wolfy that works with my RAG architecture. 
#Cons: 1. Chat history is displayed but not retained as context. 
#      2. Very Slow.


import streamlit as st
import os
import tempfile
from streamlit_chat import message
from langchain_community.chat_models import ChatOllama
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from dotenv import load_dotenv


hide_decoration_bar_style = '''
    <style>
        header {visibility: hidden;}
    </style>
'''
st.markdown(hide_decoration_bar_style, unsafe_allow_html=True)

st.markdown("""
        <style>
               .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 0rem;
                    padding-right: 0rem;
                }
                .main .block-container {
                    max-width: 80%;
                }
                .reportview-container .main .block-container {
                    padding: 0rem;
                }
        </style>
        """, unsafe_allow_html=True)


font_import = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');

html, body, [class*="sidebar-"], [class*="block-container-"], .stButton>button {
    font-family: 'Roboto', sans-serif;
}
</style>
"""

st.markdown(font_import, unsafe_allow_html=True)

ai_avatar_url = r"C:\Users\sdakshit\Documents\RAG\Llama\botlogo1.jpg"
avatar_url= r"C:\Users\sdakshit\Documents\RAG\Llama\logo.jpg"
user_avatar_url = r"C:\Users\sdakshit\Documents\RAG\Llama\botlogo.png"


col1, col2, col3, col4 = st.columns([2, 2, 4,50])
with col2:
    st.image(avatar_url, width=125)
with col4:
    st.markdown("""
    <div style='text-align: center;'>
        <h1 style='display: inline; margin: 0; padding: 0;'>Talon</h1><span style='font-size: 15px;'> -- Your Virtual Teaching Assistant for COSC 4336</span>
    </div>
    """, unsafe_allow_html=True)

divider_color = "#0096d6" 

st.markdown(f"<hr style='border-top: 2px solid {divider_color};'/>", unsafe_allow_html=True)


load_dotenv()


def initialize_session_state():
    if "history" not in st.session_state:
        st.session_state["history"] = []

    if "assistant" not in st.session_state:
        st.session_state["assistant"] = [
            "Hello Patriots, how can I help you today?"
        ]
    if "user" not in st.session_state:
        st.session_state["user"] = ["Can you help me?"]



def conversation_chat(query, chain, history):
    result = chain({
        "question": query,
        "chat_history": history
    })
    history.append((query, result["answer"]))
    return result["answer"]



def display_chat_history(chain):

        with st.chat_message("assistant", avatar =ai_avatar_url):
                st.write("Hello Patriot's', how can I help you today?")

        if "messages" not in st.session_state:
            st.session_state.messages = []


        for message in st.session_state.messages:
         with st.chat_message(message["role"], avatar=user_avatar_url if message["role"] == "user" else ai_avatar_url):
            st.markdown(message["content"]) 

        if prompt := st.chat_input("Hello Patriots, how can I help you today?"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user", avatar=user_avatar_url):
                st.markdown(prompt)


            output = conversation_chat(
                    query=prompt,
                    chain=chain,
                    history=st.session_state["history"]
                )

            with st.chat_message("assistant",  avatar =ai_avatar_url):
                st.write(output)

                st.session_state.messages.append({"role": "assistant", "content": output})




def create_conversational_chain(vector_store):

    ollama_llm= "gemma"

    model_local = ChatOllama(model=ollama_llm)

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

    template= """ You are a Teaching Assistant assisstant so answer the questions with context to Dr. Sagnik Dakshit and his Software Development class \
    Do not ask any questions in your response.\
    If you do not know the answer, say that you don't know. \
    Do not hallucinate. \
    Context: {context}
    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    chain = ConversationalRetrievalChain.from_llm(
        llm=model_local,
        chain_type="stuff",
        retriever=vector_store.as_retriever(
            search_type= "similarity_score_threshold",
            search_kwargs={
                "k" :3,  
                "score_threshold": 0.5 
            }),
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    return chain


def main():
    
    initialize_session_state()

    folder_path= r"C:\Users\sdakshit\Documents\RAG\Llama\Data"

    # Define the path to a PDF file
    pdf_path= (r"C:\Users\sdakshit\Documents\RAG\Llama\Data\merged.pdf")
    pdf_data= PyPDFLoader(file_path = pdf_path).load()
            
    embedding= SentenceTransformerEmbeddings(model_name= "all-MiniLM-L6-v2")


    vector_store = Chroma.from_documents(
            documents=pdf_data,
            embedding=embedding,
            persist_directory="chroma_store"
        )


    chain = create_conversational_chain(vector_store=vector_store)

    display_chat_history(chain=chain)


if __name__ == "__main__":
    main()
import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain_groq import ChatGroq
from langchain.prompts import ChatPromptTemplate


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


st.title('Conversational RAG using Objectbox and Groq with Retrival Chain')

system_prompt = ('''
    - You are an assistant for question-answering tasks. 
    - Use the following pieces of retrieved context to answer the question. 
    - If you don't know the answer, say that you don't know. 
    - Use three sentences maximum and keep the answer concise.
    \n\n
    {context}'''
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

llm = ChatGroq(model='llama3-8b-8192')

st.session_state.embedding = HuggingFaceEmbeddings()
st.session_state.loader = PyPDFDirectoryLoader('./data')
st.session_state.doc = st.session_state.loader.load()
st.session_state.split_doc = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(st.session_state.doc)
st.session_state.db = FAISS.from_documents(st.session_state.split_doc,st.session_state.embedding)
st.session_state.retriver = st.session_state.db.as_retriever()


prompt1 = st.text_input("Enter your question")

if prompt1:
    stuff_doc = create_stuff_documents_chain(prompt=prompt,llm=llm)
    retriver_chain = create_retrieval_chain(st.session_state.retriver,stuff_doc)
    response = retriver_chain.invoke({'input':prompt1})
    st.write(response['answer'])


import streamlit as st
import os
from dotenv import load_dotenv
from langchain.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_community.tools import WikipediaQueryRun,ArxivQueryRun
from langchain.utilities import WikipediaAPIWrapper, ArxivAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain import hub


load_dotenv()

os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')


st.title("Advance RAG using Chatgroq with Llama model")

llm = ChatGroq(model='llama3-8b-8192')


#Designing the prompt
prompt = hub.pull("hwchase17/openai-functions-agent")


def create_tools():
    if "tools" not in st.session_state:

        st.session_state.embeddings = HuggingFaceEmbeddings()
        st.session_state.wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1,doc_content_chars_max=200))
        st.session_state.arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1,doc_content_chars_max=200))
        st.session_state.loader = WebBaseLoader('https://docs.smith.langchain.com/')
        st.session_state.doc = st.session_state.loader.load()
        st.session_state.split_doc = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=200).split_documents(st.session_state.doc)
        st.session_state.db = FAISS.from_documents(st.session_state.split_doc,st.session_state.embeddings).as_retriever()
        st.session_state.web_tool = create_retriever_tool(st.session_state.db,'web tool','responsible for retriving from web')
        st.session_state.loader1 = PyPDFLoader('attention.pdf')
        st.session_state.doc1 = st.session_state.loader1.load()
        st.session_state.split_doc1 = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100).split_documents(st.session_state.doc1)
        st.session_state.db1 = FAISS.from_documents(st.session_state.split_doc1,st.session_state.embeddings).as_retriever()
        st.session_state.pdf_tool = create_retriever_tool(st.session_state.db1,'pdf tool','responsible for retriving from pdf')
        st.session_state.tools = [st.session_state.wiki,st.session_state.arxiv,st.session_state.web_tool,st.session_state.pdf_tool]
    

prompt1= st.text_input('enter your question')


if st.button('Create Tools'):
    create_tools()
    st.write('Tools are ready')

import time
if prompt1:
    agents = create_tool_calling_agent(tools=st.session_state.tools,llm=llm,prompt=prompt)
    agents_exe = AgentExecutor(agent=agents,tools=st.session_state.tools,verbose=1)
    start = time.process_time()
    response = agents_exe.invoke({'input':prompt1})
    print("Response time :",time.process_time()-start)
    st.write(response['output'])



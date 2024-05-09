import os
import streamlit as st
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain.embeddings import OllamaEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS 
from langchain.chains.combine_documents import create_stuff_documents_chain
import time
# create_stuff_document_chain is an object used in retrieval chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

load_dotenv()
groq_api_key = os.environ["GROQ_API_KEY"]


if "vector" not in st.session_state:
    st.session_state.embeddings = OpenAIEmbeddings()
    st.session_state.loader     = PyPDFLoader("Attention.pdf")
    st.session_state.docs       = st.session_state.loader.load()
    st.session_state.splitter   = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap=200)
    st.session_state.split_docs = st.session_state.splitter.split_documents(st.session_state.docs[:50])
    st.session_state.db         = FAISS.from_documents(st.session_state.split_docs,st.session_state.embeddings)

st.header("Groq Inferencing Engine")
llm = ChatGroq(name="mixtral-8x7b-32768",
               groq_api_key=groq_api_key)


prompt = ChatPromptTemplate.from_template(
"""
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question
<context>
{context}
<context>
Questions:{input}
"""
)


doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
retriever = st.session_state.db.as_retriever()
retrieval_chain = create_retrieval_chain(retriever,doc_chain)

query = st.text_input("input query")
if query:
    start = time.process_time()
    response = retrieval_chain.invoke({"input":query})
    print("time taken :",time.process_time()-start)
    st.write(response['answer'])

    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
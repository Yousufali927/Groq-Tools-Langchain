import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS 
import time
import requests
# create_stuff_document_chain is an object used in retrieval chain

from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import RetrievalQA 

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv("HUGGINGFACEHUB_API_TOKEN")
inference_api_key = os.getenv("HF_inference_api_key")
API_KEY = os.getenv("HUGGINGFACE_API_TOKEN")

loader = PyPDFLoader("Attention.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
split_docs = text_splitter.split_documents(documents)

final_docs = split_docs[:50]

HF_embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=inference_api_key, model_name="sentence-transformers/all-MiniLM-l6-v2"
)

db = FAISS.from_documents(final_docs,HF_embeddings)

prompt = """
Use the following piece of context to answer the question asked.
Please try to provide the answer only based on the context

{context}
Question:{query}

Helpful Answers:
"""
MODEL_ENDPOINT = 'https://api-inference.huggingface.co/models/mistralai/Mistral-7B-v0.1'

st.header("HuggingFace Inferencer")
query = st.text_input("input query")

headers = {
    "Authorization": f"Bearer {API_KEY}"
}


if query:
    start = time.process_time()
    context = db.similarity_search(query)
    final_prompt = prompt.format(context=context,query=query)
    full_input = final_prompt
    data = {
    "inputs": full_input
    }
    response = requests.post(MODEL_ENDPOINT, headers=headers, json=data)
    end = time.process_time()-start
    print("time taken to get the response is:",end)
    print(response.json())
    st.write(response.json())



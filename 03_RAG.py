from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama 
import os 
from dotenv import load_dotenv
load_dotenv()

os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")


# can use WebBaseLoader to load web pages but here we're going to read a pdf
loader = PyPDFLoader("attention.pdf")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
split_docs = text_splitter.split_documents(documents)

# can also use Ollama Embeddings if you dont have openai API Access
db = FAISS.from_documents(split_docs,OpenAIEmbeddings())

####### reading and storing the pdf file ########

####### chain to talk with the llm #########

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
import streamlit as st

# if you dont have access to openai api, download llama2 from Ollama(check documentation online)
# use model = Ollama(model="llama2")

model = ChatOpenAI(model = "gpt-3.5-turbo")
output_parser = StrOutputParser()

prompt = ChatPromptTemplate.from_messages(
    [
        ("system","You are a help assistant. that answers the user's query based on the given context, if the context doesnt contain any related info, just answer `I DONT KNOW!!!!` "),
        ("user", "Query:{query},context:{context}")
    ]
)

chain = prompt|model|output_parser

#### streamlit app ######

st.header("RAG using OpenAI/llama2(Ollama)")

query = st.text_input("input query")
if query:
    retrieved = db.similarity_search(query)
    context = retrieved[0].page_content
    answer = chain.invoke({"query":query,"context":context})
    st.write(answer)
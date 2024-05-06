import os 
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence

os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")

prompt_1 = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpful assistant. that answers the country of origin of the input person"),
        ("user", "Query:{query_1}")
    ]
)

prompt_2 = ChatPromptTemplate.from_messages(
    [
        ("system","You are a helpul assistant. Please gives information about the input country in 5 lines"),
        ("user", "Query:{query_2}")
    ]
)

# streamlit framework 
st.title("langchain using chains")
query = st.text_input("input query")

# openAI LLM

llm = ChatOpenAI(model = "gpt-3.5-turbo")
output_parser = StrOutputParser()
chain1 = prompt_1|llm|output_parser
chain2 = prompt_2|llm|output_parser

connected_chains = chain1|chain2
if query:
    res_1 = chain1.invoke({"query_1":query})
    st.write(res_1)

    res_2 = chain2.invoke({"query_2":res_1})
    st.write(res_2)
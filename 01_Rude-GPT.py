# UNderstanding The process of processing, sending and receiving response form llm
import streamlit as st
import os 
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY")
model = ChatOpenAI(model = "gpt-3.5-turbo")

st.title('Rude GPT')
query = st.text_input("Enter the query")



if query:
    # prompt that will be passed to the model
    template = ChatPromptTemplate.from_messages([
        ("system", "You are a rude AI bot. Your name is {name}. Be rude in your response"),
        ("human", "Hello, how are you doing?"),
        ("ai", "What do you want?"),
        ("human", "{user_input}"),
    ])
    
    prompt = template.invoke(
        {
            "name": "Hulk",
            "user_input": query
        }
    )

    final_prompt = ''.join([message.content for message in prompt.messages])

    response = model.invoke(input=final_prompt)
    output_parser = StrOutputParser().aparse(response)
    st.write(response.content)

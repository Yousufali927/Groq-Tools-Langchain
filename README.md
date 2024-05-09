# Groq-Tools-Langchain

1. Rude_GPT
2. Handling-chains
3. RAG
4. Groq_RAG
---------------------------------

### 1. Rude_GPT
This version of AI Chatbot helps understand the importance of the prompt passed to the llm. It is built using langchain, OpenAIAPI, Streamlit. It gives rude responses to all the queries and is quite aggressive, it is kinda funny tbh🙂, check it out for yourself. 

### 2. Handling-chains
In this code we build and understand the working of chains in langchain, instead of simply using the Sequential chain, we handle the output of each chain manually to understand the processs going on inside chains like SimpleSequentailChain and SequentialChain.

Uses chains from langchain, OpenAIAPI, streamlit

### 3. RAG
Building a streamlit app implementing a RAG(Retrieval Augmented Generation) system using Langchain. We read a pdf "attention.pdf" from our directory, make the embeddings using OpenAIEmbeddings, store them in the FAISS database, and instead of using a simple retrival_chain we do handle the output ourselves for better understanding of the variables, outputs and logic involved

### 4. Groq
Groq's Inferencing engine is the best in the market by a long shot and free available access to inference using provided opensource models means you gotta check it out.

The code implements a RAG system (yes again🙂), the embeddings and DB are present in session_state so they don't have to reload everytime we run script. and we don't handle the output of chains manually this time, instead we use the stuff_doc_chain and the retrival_chain, streamling the process of getting the output.

Make sure to get the Groq API Access from here :
https://wow.groq.com/

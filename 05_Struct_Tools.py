import requests
import streamlit as st
from textblob import TextBlob
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
load_dotenv()


# Initialize the ChatOpenAI client with your API key and desired model
chat_client = ChatOpenAI(api_key='your_openai_api_key_here',model="gpt-3.5-turbo")

# Define the function to get news articles
def get_news(topic):
    news_api_key = 'd4985458f547455eaa21c858eee02983'
    base_url = "your_news_api_key_here"
    complete_url = f"{base_url}?q={topic}&apiKey={news_api_key}"
    response = requests.get(complete_url)
    articles = response.json().get('articles', [])
    return articles

# Define the function to analyze sentiment
def analyze_sentiment(text):
    # Check if the text is None
    if text is None:
        return 0  # Return a neutral sentiment score
    analysis = TextBlob(text)
    return analysis.sentiment.polarity


# Define the function to get facts from GPT-3.5-turbo using ChatOpenAI
def get_facts(topic):
    response = chat_client.invoke(f"Tell me some interesting facts about {topic}:")
    return response

# Define the structured tool
def structured_tool(topic):
    news_articles = get_news(topic)
    facts = get_facts(topic)
    
    # Analyze sentiment for each article
    sentiments = [analyze_sentiment(article['description']) for article in news_articles]
    average_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0
    
    # Create a structured report
    report = {
        "topic": topic,
        "facts": facts,
        "news_sentiment": "Positive" if average_sentiment > 0 else "Negative",
        "articles": [{"title": article['title'], "url": article['url']} for article in news_articles]
    }
    return report

# Streamlit app
st.title('Comprehensive Topic Report')

topic = st.text_input('Enter a topic to generate a report:', '')

if topic:
    with st.spinner('Generating report...'):
        report = structured_tool(topic)
        st.write('Facts:')
        st.write(report['facts'].content)
        st.write('News Sentiment:')
        st.write(report['news_sentiment'])
        st.write('Related Articles:')
        n = 1
        for article in report['articles']:
            if n<5:
                st.write(f"[{article['title']}]({article['url']})")
                n+=1








































from langchain.agents import Tool
from langchain_core.tools import tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROK_API_KEY"),
)

@tool
def analyze_sentiment(text: str) -> str:
    """
    Analyzes the sentiment of a given text (Positive, Negative, or Neutral) using LLM.
    """
    prompt = f"""
    Classify the sentiment of the following text as Positive, Negative, or Neutral. 
    Just return one word.

    Text: \"{text}\"
    """
    try:
        return groq_llm.invoke(prompt)
    except Exception as e:
        return f"Error: {e}"

text_sentiment_tool = Tool(
    name="Sentiment Analyzer",
    func=analyze_sentiment,
    description="Analyzes sentiment of text as Positive, Negative, or Neutral using the LLM."
)

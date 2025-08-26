import os
from dotenv import load_dotenv
load_dotenv()

from langchain.agents import initialize_agent, AgentType
from langchain_groq import ChatGroq
# from tools.google_search import google_search_tool
from tools.calculator import calculator_tool
from tools.file_writer import file_writer_tool
from tools.text_sentiment import text_sentiment_tool
from tools.rag import rag_tool

tools = [calculator_tool, file_writer_tool, text_sentiment_tool, rag_tool]

groq_llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("GROK_API_KEY"),
)

agent = initialize_agent(
    tools=tools,
    llm=groq_llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

if __name__ == "__main__":
    print("\n Welcome to #in Chatbot! Type 'exit' to quit.\n")
    while True:
        user_input = input(" You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        try:
            response = agent.run(user_input)
            print("Bot:", response)
        except Exception as e:
            print("Error:", e)

import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from langchain.agents import Tool
from langchain_core.tools import tool, BaseTool

search_tool = GoogleSerperAPIWrapper(api_key=os.getenv("SERPER_API_KEY"))
@tool
def google_search(text: str) -> str:
    """
    Searches Google for the given text.

    Args:
        text (str): The text to search for.

    Returns:
        str: The search results.
    """
    return search_tool.run(text)
    
google_search_tool = Tool(
     name="Google Search",
     func=google_search,
     description="Search Google for real-time information."
)
result = google_search_tool.run("Who is Virat Kohli?")
print(result)

@tool
def calculator(inputs: str):
    """
    Performs basic arithmetic calculations. Input should be a mathematical expression.

    Args:
        inputs (str): The text to evaluate the expression.

    Returns:
        str: The calculated results.
    """
    try:
        return str(eval(inputs)) 
    except Exception as e:
        return f"Error: {e}"

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Performs basic arithmetic calculations. Input should be a mathematical expression."
)

from langchain.agents import Tool
from langchain_core.tools import tool

@tool
def calculator(inputs: str) -> str:
    """Performs basic arithmetic calculations. Input must be a math expression."""
    try:
        return str(eval(inputs))
    except Exception as e:
        return f"Error: {e}"

calculator_tool = Tool(
    name="Calculator",
    func=calculator,
    description="Performs basic arithmetic calculations. Input should be a math expression."
)
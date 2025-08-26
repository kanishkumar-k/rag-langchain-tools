from langchain.agents import Tool
from langchain_core.tools import tool

@tool
def write_to_file(content: str) -> str:
    """Writes the input text to a file called output.txt and returns a success message."""
    try:
        with open("output.txt", "w") as f:
            f.write(content +" ")
        return "Text written to output.txt"
    except Exception as e:
        return f"Error writing to file: {e}"

file_writer_tool = Tool(
    name="File Writer",
    func=write_to_file,
    description="Writes user-provided content to output.txt"
)

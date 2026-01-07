```python
"""
Tool Test – Calculator (LangChain)

This script demonstrates how to define and execute a LangChain tool.

---------------------------------
Setup & Run Instructions
---------------------------------

1. Create a virtual environment:
   python3 -m venv venv

2. Activate the virtual environment:
   source venv/bin/activate

3. Install dependencies:
   pip install langchain

4. Run the script:
   python3 tool_test.py
"""


from langchain.tools import tool

@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

if __name__ == "__main__":
    print("Tool name:", calculator.name)
    print("Tool description:", calculator.description)
    print("Tool args schema:")
    print(calculator.args_schema.model_json_schema())

    print("\nTool execution result:")
    print(calculator.invoke({"a": 5, "b": 6}))


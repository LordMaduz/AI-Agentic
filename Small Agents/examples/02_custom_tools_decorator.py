"""
Example 02: Custom Tools with @tool Decorator

Demonstrates how to create custom tools using the @tool decorator.
This is the simplest way to extend agent capabilities with your own functions.

Features:
- @tool decorator: Convert functions into agent-callable tools
- CodeAgent: Agent that can write and execute Python code
- Custom tool with typed arguments and docstring
"""

from smolagents import CodeAgent, InferenceClientModel, tool


@tool
def restaurant_finder(cuisine: str) -> str:
    """
    This tool returns the highest-rated restaurant for a given cuisine type.

    Args:
        cuisine: The type of cuisine to search for (e.g., Italian, Japanese, Mexican).
    """
    # Simulated database of restaurants and their ratings
    restaurants = {
        "italian": {"name": "Bella Napoli", "rating": 4.9},
        "japanese": {"name": "Sakura Garden", "rating": 4.8},
        "mexican": {"name": "Casa del Sol", "rating": 4.7},
        "indian": {"name": "Spice Route", "rating": 4.8},
        "french": {"name": "Le Petit Bistro", "rating": 4.6},
    }

    cuisine_lower = cuisine.lower()
    if cuisine_lower in restaurants:
        result = restaurants[cuisine_lower]
        return f"{result['name']} (Rating: {result['rating']})"

    return "No restaurant found for this cuisine type."


agent = CodeAgent(tools=[restaurant_finder], model=InferenceClientModel())

# Run the agent to find the best restaurant
result = agent.run(
    "Can you find me the highest-rated Italian restaurant for dinner tonight?"
)

print(result)

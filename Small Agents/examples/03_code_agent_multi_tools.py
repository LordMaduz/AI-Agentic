"""
Example 03: CodeAgent with Multiple Tools

Demonstrates CodeAgent with a combination of built-in and custom tools.
Shows two ways to create custom tools: @tool decorator and Tool class.

Features:
- DuckDuckGoSearchTool: Web search via DuckDuckGo
- VisitWebpageTool: Fetch and parse webpage content
- @tool decorated functions: Simple custom tools
- Tool class extension: Complex custom tools with defined inputs/outputs
- FinalAnswerTool: Explicitly mark final answers
- max_steps and verbosity_level configuration

Requirements:
    pip install duckduckgo-search
"""

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    FinalAnswerTool,
    InferenceClientModel,
    Tool,
    tool,
    VisitWebpageTool,
)


@tool
def suggest_menu(occasion: str) -> str:
    """
    Suggests a menu based on the occasion.
    Args:
        occasion: The type of occasion for the event (casual, formal, birthday, etc.).
    """
    menus = {
        "casual": "Finger foods, pizza, chips with dips, and refreshing drinks.",
        "formal": "3-course dinner: appetizer salad, main course with wine, and dessert.",
        "birthday": "Custom cake, finger sandwiches, fruit platter, and party snacks.",
        "brunch": "Eggs benedict, fresh pastries, mimosas, and fruit bowls.",
        "bbq": "Grilled burgers, hot dogs, corn on the cob, and coleslaw.",
    }
    return menus.get(occasion.lower(), "Custom menu based on your preferences.")


@tool
def catering_service_finder(location: str) -> str:
    """
    This tool returns the highest-rated catering service in a given location.

    Args:
        location: The city or area to search for catering services.
    """
    # Simulated database of catering services by location
    services = {
        "new york": {"name": "NYC Elite Catering", "rating": 4.9},
        "los angeles": {"name": "LA Gourmet Events", "rating": 4.8},
        "chicago": {"name": "Windy City Catering", "rating": 4.7},
        "default": {"name": "Local Best Catering", "rating": 4.5},
    }

    location_lower = location.lower()
    result = services.get(location_lower, services["default"])
    return f"{result['name']} (Rating: {result['rating']})"


class PartyThemeGenerator(Tool):
    """
    Custom Tool class demonstrating the more complex way to define tools.
    Use this approach when you need fine-grained control over inputs/outputs.
    """

    name = "party_theme_generator"
    description = """
    This tool suggests creative party theme ideas based on a category.
    It returns a unique party theme with decoration and activity suggestions."""

    inputs = {
        "category": {
            "type": "string",
            "description": "The type of party theme (e.g., 'retro', 'tropical', 'elegant', 'movie night').",
        }
    }

    output_type = "string"

    def forward(self, category: str):
        themes = {
            "retro": "80s Throwback Party: Neon decorations, disco ball, vintage arcade games, and synth-pop playlist.",
            "tropical": "Hawaiian Luau: Tiki torches, tropical flowers, fruity cocktails, and beach-themed games.",
            "elegant": "Gatsby Glamour: Art deco decorations, jazz music, champagne tower, and black-tie dress code.",
            "movie night": "Hollywood Premiere: Red carpet entrance, popcorn bar, movie posters, and award ceremony games.",
            "garden": "Secret Garden Party: Fairy lights, floral arrangements, outdoor games, and afternoon tea.",
        }

        return themes.get(
            category.lower(),
            "Custom theme available! Try 'retro', 'tropical', 'elegant', 'movie night', or 'garden'.",
        )


# Initialize agent with multiple tools
agent = CodeAgent(
    tools=[
        DuckDuckGoSearchTool(),
        VisitWebpageTool(),
        suggest_menu,
        catering_service_finder,
        PartyThemeGenerator(),
        FinalAnswerTool(),
    ],
    model=InferenceClientModel(),
    max_steps=10,
    verbosity_level=2,
)

# Run multiple queries
agent.run("Search for the best playlist for a summer rooftop party.")
agent.run("Suggest a menu for a formal dinner party.")
agent.run("Give me a creative theme idea for a retro-themed birthday party.")

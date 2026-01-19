"""
Example 04: Single Agent with Pandas and Geographic Calculations

Demonstrates a more advanced single agent setup with custom mathematical
calculations (haversine formula) and pandas DataFrame generation.

Features:
- Custom calculation tool: Great-circle distance using haversine formula
- additional_authorized_imports: Allow pandas in agent sandbox
- planning_interval: Control how often agent replans
- InferenceClientModel with specific model and provider
- Complex multi-step reasoning task

Requirements:
    pip install pandas duckduckgo-search pillow
"""

import math
from typing import Optional, Tuple

from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    tool,
    VisitWebpageTool,
)


@tool
def calculate_flight_time(
    origin_coords: Tuple[float, float],
    destination_coords: Tuple[float, float],
    cruising_speed_kmh: Optional[float] = 850.0,
) -> float:
    """
    Calculate the estimated flight time between two points on Earth using great-circle distance.

    Args:
        origin_coords: Tuple of (latitude, longitude) for the starting point
        destination_coords: Tuple of (latitude, longitude) for the destination
        cruising_speed_kmh: Optional cruising speed in km/h (defaults to 850 km/h for commercial flights)

    Returns:
        float: The estimated travel time in hours

    Example:
        >>> # New York (40.7128° N, 74.0060° W) to London (51.5074° N, 0.1278° W)
        >>> result = calculate_flight_time((40.7128, -74.0060), (51.5074, -0.1278))
    """

    def to_radians(degrees: float) -> float:
        return degrees * (math.pi / 180)

    # Extract coordinates
    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    # Earth's radius in kilometers
    EARTH_RADIUS_KM = 6371.0

    # Calculate great-circle distance using the haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    # Add 10% to account for non-direct routes and air traffic
    actual_distance = distance * 1.1

    # Calculate flight time (add 1 hour for takeoff/landing procedures)
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    return round(flight_time, 2)


# Configure model with specific provider
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    provider="together",
)

task = """Find popular tourist destinations in Europe, calculate the flight time from New York (40.7128° N, 74.0060° W) to each destination, and return them as a pandas dataframe.
Include at least 5 destinations with their coordinates and flight times."""

# Initialize agent with pandas support
travel_agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_flight_time],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

# Set planning interval for better task organization
travel_agent.planning_interval = 4

detailed_report = travel_agent.run(
    f"""
You're a travel research assistant. You help users plan trips by gathering destination information.
Don't hesitate to search for multiple queries to gather comprehensive data.
For each destination, find accurate coordinates to calculate flight times.

{task}
"""
)

print(detailed_report)

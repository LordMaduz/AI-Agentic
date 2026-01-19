"""
Example 05: Multi-Agent Orchestration with Vision Validation

Demonstrates hierarchical multi-agent architecture where a manager agent
orchestrates worker agents. Includes vision-based validation of results.

Features:
- Managed agents: Hierarchical agent architecture
- OpenAIServerModel: GPT-4o for multimodal/vision tasks
- final_answer_checks: Validation functions for agent outputs
- Vision-based reasoning: Analyze generated plots with GPT-4o
- Plotly visualization: Generate interactive maps
- Multiple model providers in same workflow

Requirements:
    pip install pandas duckduckgo-search pillow plotly geopandas shapely numpy openai kaleido
"""

import math
import os
from typing import Optional, Tuple

from PIL import Image
from smolagents import (
    CodeAgent,
    DuckDuckGoSearchTool,
    InferenceClientModel,
    OpenAIServerModel,
    tool,
    VisitWebpageTool,
)
from smolagents.utils import encode_image_base64, make_image_url


def check_reasoning_and_plot(final_answer, agent_memory):
    """
    Validation function that uses GPT-4o vision to verify the agent's output.
    This function is called automatically when the agent produces a final answer.
    """
    multimodal_model = OpenAIServerModel("gpt-4o", max_tokens=8096)
    filepath = "saved_map.png"
    assert os.path.exists(filepath), "Make sure to save the plot under saved_map.png!"
    image = Image.open(filepath)
    prompt = (
        f"Here is a user-given task and the agent steps: {agent_memory.get_succinct_steps()}. Now here is the plot that was made."
        "Please check that the reasoning process and plot are correct: do they correctly answer the given task?"
        "First list reasons why yes/no, then write your final decision: PASS in caps lock if it is satisfactory, FAIL if it is not."
        "Don't be harsh: if the plot mostly solves the task, it should pass."
        "To pass, a plot should be made using px.scatter_map and not any other method (scatter_map looks nicer)."
    )
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt,
                },
                {
                    "type": "image_url",
                    "image_url": {"url": make_image_url(encode_image_base64(image))},
                },
            ],
        }
    ]
    output = multimodal_model(messages).content
    print("Feedback: ", output)
    if "FAIL" in output:
        raise Exception(output)
    return True


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

    lat1, lon1 = map(to_radians, origin_coords)
    lat2, lon2 = map(to_radians, destination_coords)

    EARTH_RADIUS_KM = 6371.0

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.asin(math.sqrt(a))
    distance = EARTH_RADIUS_KM * c

    actual_distance = distance * 1.1
    flight_time = (actual_distance / cruising_speed_kmh) + 1.0

    return round(flight_time, 2)


# Worker agent for web research
model = InferenceClientModel(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    provider="together",
)

web_agent = CodeAgent(
    model=model,
    tools=[DuckDuckGoSearchTool(), VisitWebpageTool(), calculate_flight_time],
    additional_authorized_imports=["pandas"],
    max_steps=20,
)

# Manager agent that orchestrates the web_agent
manager_agent = CodeAgent(
    model=InferenceClientModel(
        "deepseek-ai/DeepSeek-R1", provider="together", max_tokens=8096
    ),
    tools=[calculate_flight_time],
    managed_agents=[web_agent],  # Worker agents managed by this agent
    additional_authorized_imports=[
        "geopandas",
        "plotly",
        "shapely",
        "json",
        "pandas",
        "numpy",
    ],
    planning_interval=5,
    verbosity_level=2,
    final_answer_checks=[check_reasoning_and_plot],  # Vision-based validation
    max_steps=15,
)

# Visualize agent hierarchy
manager_agent.visualize()

# Run the orchestrated task
manager_agent.run(
    """
Find popular coffee shop chains headquarters locations around the world and major tech company headquarters.
Calculate the flight time from San Francisco (37.7749° N, 122.4194° W) to each location.
You need at least 6 points in total.

Represent this as a spatial map of the world, with the locations represented as scatter points
with a color that depends on the flight time, and save it to saved_map.png!

Here's an example of how to plot and return a map:
import plotly.express as px
df = px.data.carshare()
fig = px.scatter_map(df, lat="centroid_lat", lon="centroid_lon", text="name", color="peak_hour", size=100,
     color_continuous_scale=px.colors.sequential.Magma, size_max=15, zoom=1)
fig.show()
fig.write_image("saved_image.png")
final_answer(fig)

Never try to process strings using code: when you have a string to read, just print it and you'll see it.
"""
)

# Access the generated figure from agent state
manager_agent.python_executor.state["fig"]

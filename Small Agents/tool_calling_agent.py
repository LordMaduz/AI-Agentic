from smolagents import ToolCallingAgent, WebSearchTool, InferenceClientModel

agent = ToolCallingAgent(tools=[WebSearchTool()], model=InferenceClientModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
from agents import Agent, WebSearchTool, ModelSettings, function_tool
from tavily import TavilyClient
from dotenv import load_dotenv
import os
from model import gemini_model

INSTRUCTIONS = (
    "You are a research assistant. Given a search term, you search the web for that term and "
    "produce a concise summary of the results. The summary must 2-3 paragraphs and less than 300 "
    "words. Capture the main points. Write succintly, no need to have complete sentences or good "
    "grammar. This will be consumed by someone synthesizing a report, so its vital you capture the "
    "essence and ignore any fluff. Do not include any additional commentary other than the summary itself."
)

@function_tool
def search_web(query: str) -> str:
    """
    Searches the web using Tavily API and returns the top result.
    Args:
        query (str): The search query.
    Returns:
        str: The title and URL of the top search result.
    """
    load_dotenv()
    api_key = os.getenv("TAVILY_API_KEY")
    client = TavilyClient(api_key=api_key)
    results = client.search(query=query, limit=3)
    return str(results)

search_agent = Agent(
    name="Search agent",
    instructions=INSTRUCTIONS,
    # tools=[WebSearchTool(search_context_size="low")],
    tools=[search_web],
    model=gemini_model,
    model_settings=ModelSettings(tool_choice="required"),
)
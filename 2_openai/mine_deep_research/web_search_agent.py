from agents import Agent, trace, Runner
from model import gemini_model
from textwrap import dedent
from agents.model_settings import ModelSettings
from planner_agent import planner_agent
from search_agent import search_agent
from typing import Literal
from dataclasses import dataclass
import asyncio

@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]

evaluator = Agent[None](
    name="evaluator",
    instructions=dedent("""
    You are an expert evaluator. You will be given a set of search results and you need to evaluate if they satisfy the search query.
    If more information is needed, you should provide feedback on what to search for next, and fail the search.
    If the search results are satisfactory, you should provide a score of "pass" and a summary of the results.
    If improvements are needed, you should provide a score of "needs_improvement" and a summary of the results.
    If the initial request is too vague, you should recommend new searches to refine the query and fail the search.
    Here are the rules you should always follow to solve your task:
    1. Evaluate the initial search request vs the search results.
    2. Focus on completeness and relevance of the search results.
    3. Provide feedback on what to search for next if more information is needed. 
    """),
    output_type=EvaluationFeedback,
    model=gemini_model,
    model_settings=ModelSettings(tool_choice="required", temperature=0.0, max_tokens=400),
)

web_search_tool = [
    planner_agent.as_tool(
        tool_name="plan_searches",
        tool_description="Plan a set of web searches to perform",
    ),
    search_agent.as_tool(
        tool_name="search",
        tool_description="Search the web for information",
    ),
    evaluator.as_tool(
        tool_name="evaluate",
        tool_description="Evaluate the search results vs the search query and provide feedback",
    ),
]

tool_name_map = "\n".join([f"{tool.name}: {tool.description}" for tool in web_search_tool])

prompt = dedent(f"""
    You are a web search agent. You will be given a task to solve as best you can.
    You should use all the tools available to you to solve the task.
    Here are the tools you can use:
    {tool_name_map}
    You should operate in a cycle using your tools in the following order:
    1. Plan the searches to perform
    2. Perform the searches
    3. Evaluate the search results
    4. Refine the searches based on the evaluation
    5. Repeat the cycle until you have enough information to solve the task
    """)

web_search_agent = Agent(
    name="WebSearchAgent",
    instructions=prompt,
    model=gemini_model,
    model_settings=ModelSettings(tool_choice="required"),
    tools=web_search_tool,
)

async def main():

    with trace("Joke workflow"): 
        first_result = await Runner.run(web_search_agent, "What is the latest news about AI?")
        print(f"Search results: {first_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
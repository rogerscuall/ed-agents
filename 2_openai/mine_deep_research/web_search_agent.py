from agents import Agent, trace, Runner, function_tool
from model import gemini_model
from textwrap import dedent
from agents.model_settings import ModelSettings
from planner_agent import planner_agent, WebSearchPlan
from search_agent import search_agent
from typing import Literal
from dataclasses import dataclass
import asyncio

call = 0

@dataclass
class EvaluationFeedback:
    feedback: str
    score: Literal["pass", "needs_improvement", "fail"]

@function_tool
async def evaluate_search_results(
    initial_request: str,
    search_query: WebSearchPlan,
    search_results: str) -> EvaluationFeedback:
    """
    Evaluate the search results vs the search query and provide feedback
    Args:
        initial_request: The initial request to evaluate
        search_query: The search query to evaluate
        search_results: The search results to evaluate
    Returns:
        EvaluationFeedback: The evaluation feedback
    """
    global call
    call += 1
    evaluator_agent = Agent(
        name="evaluator",
        instructions=dedent(f"""
        You are an expert evaluator. You will be given a set of search results and you need to evaluate if they satisfy the search query.
        Part of your job is to be creative and think outside the box while providing feedback.
        If more information is needed, you should provide feedback on what to search for next, and fail the search.
        This is the {call}th time you are called. You should avoid providing "pass" score if this is the first time you are called.
        If the search results are satisfactory, you should provide a score of "pass" and a summary of the results.
        If improvements are needed or is one of the first time, you should provide a score of "needs_improvement" and a summary of the results.
        If the initial request is too vague, you should recommend new searches to refine the query and fail the search.
        Here are the rules you should always follow to solve your task:
        1. Evaluate the initial search request vs the search results.
        2. Focus on completeness and relevance of the search results.
        3. Provide feedback on what to search for next if more information is needed. 
        The initial request is: {initial_request}
        The search query is: {search_query}
        The search results are: {search_results}
        """),
        output_type=EvaluationFeedback,
        model=gemini_model,
        model_settings=ModelSettings(tool_choice="required", temperature=0.9, max_tokens=400))

    result = await Runner.run(evaluator_agent, f"Good luck!", max_turns=30)
    return result.final_output_as(EvaluationFeedback)


web_search_tool = [
    planner_agent.as_tool(
        tool_name="plan_searches",
        tool_description="Plan a set of web searches to perform",
    ),
    search_agent.as_tool(
        tool_name="search",
        tool_description="Search the web for information",
    ),
    evaluate_search_results
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
    model_settings=ModelSettings(tool_choice="required", temperature=0.0),
    tools=web_search_tool,
)

async def main():
    with trace("WebSearch") as my_trace: 
        first_result = await Runner.run(web_search_agent, "'What are the latest changes for tax code in 2024?'")
        print(f"Search results: {first_result.final_output}")

if __name__ == "__main__":
    asyncio.run(main())
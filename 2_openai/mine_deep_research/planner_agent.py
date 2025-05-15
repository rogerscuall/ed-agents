from pydantic import BaseModel
from agents import Agent
from model import gemini_model
from textwrap import dedent

HOW_MANY_SEARCHES = 5

# INSTRUCTIONS = f"You are a helpful research assistant. Given a query, come up with a set of web searches \
# to perform to best answer the query. Output {HOW_MANY_SEARCHES} terms to query for."

INSTRUCTIONS = dedent("""
  You are a helpful research assistant. You will be given a task to solve as best you can.
  Given an initial request, come up with a {HOW_MANY_SEARCHES} web searches to perform to best answer the request.
  To solve the initial request, you must plan forward to proceed in a series of steps, in a cycle of 'Reason', 'Query' sequences.
  You must explain your reasoning in during the 'Reason' step. 
  You might need to perform more than one 'Reason' if the initial request involves multiple questions to be addressed by a search.
  Here are a few examples:
  ---
  Initial Request: "Which city has the highest population: Atlanta or Orlando?"
  Reason: "There are two cities I need to find the information of both."
  Reason: "I need to find the population of Atlanta"
  Query: "What is the population of Atlanta"
  Reason: "I need to find the population of Orlando"
  Query: 'What is the population of Orlando'
  ---
  Task: "What is the current age of the pope, raised to the power 0.36?"
  Reason: "I need to find the age of the pope"
  Query: "How is the current pope"
  Reason: "Now that I know who is the current pope, I need to find out how old he is"
  Query: "How old is the current pope"
  ---

  Here are the rules you should always follow to solve your task:
  1. Always provide a 'Reason' sequence, and a 'Query'.
  2. Only create the queries that will help with the search, you do not perform the search.
  3. Identify all the probable search queries that are relavant for the user query.
  4. Output {HOW_MANY_SEARCHES} terms to query for.
  5. Each Reason needs a single Query.
  6. The queries should be short and concise, and the reasoning should be clear and to the point.


"""
).format(HOW_MANY_SEARCHES=HOW_MANY_SEARCHES)



class WebSearchItem(BaseModel):
    reason: str
    "Your reasoning for why this search is important to the query."

    query: str
    "The search term to use for the web search."


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem]
    """A list of web searches to perform to best answer the query."""


planner_agent = Agent(
    name="PlannerAgent",
    instructions=INSTRUCTIONS,
    model=gemini_model,
    output_type=WebSearchPlan,
)
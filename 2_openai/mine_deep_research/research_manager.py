from agents import Runner, trace, gen_trace_id, Agent, ItemHelpers
from agents.model_settings import ModelSettings
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from web_search_agent import web_search_agent
from email_agent import email_agent
from openai.types.responses import ResponseTextDeltaEvent
from textwrap import dedent
from model import gemini_model

def create_prompts(tools: list):
    """ Create the prompts for the tools """
    tool_descriptions = "\n".join([f"{tool.name}: {tool.description}" for tool in tools])
    instructions = dedent(f"""
        You are a helpful research assistant. You will be given a task and you need to prepare a report and send it via email.
        You have access to the following tools:
        {tool_descriptions}
        To solve a research task, you must use the tools at your disposal to proceed in a series of steps, in a cycle of 'Thought:', 'Tool:', and 'Obervation'.
        You might need to repeat this cycle multiple times to solve the task.
        Here are a few examples:
        ---
        Task: "Which city has the highest population: Atlanta or Orlando?"
        Thought: "There are two cities I need to find the information of both."
        Tool: web_search
        Observation: "I need to find the population of Atlanta"
        Thought: "I need to find the population of Orlando"
        Tool: web_search
        Observation: 'I have found the population of Atlanta and Orlando, I compare them'
        Thought: "I have all the information I need I can prepare the report"
        Tool: write_report
        Observation: "I have written the report"
        Thought: "I need to send the report"
        Tool: send_email
        ---
        Task: "What is the latest news about AI"
        Thought: "I need to search for the latest news about AI"
        Tool: web_search
        Observation: "I have found the latest news about AI and application to medicine and finance"
        Thought: "I should search for more specific news about AI in medicine"
        Tool: web_search
        Observation: "I have found the latest news about AI in medicine"
        Thought: "I should look for more specific news about AI in finance"
        Tool: web_search
        Observation: "I have found the latest news about AI in finance"
        Thought: "I have all the information I need I can prepare the report"
        Tool: write_report
        Observation: "I have written the report"
        Thought: "I need to send the report"
        Tool: send_email
        ---

        Here are the rules you should always follow to solve your task:
        1. Always provide a 'Thought', 'Tool', 'Obersvation'.
        2. Once initial thoughs and searches are done, be open to iterate and refine your search using the new information you find.
        3. Identify all the probable search queries that are relavant for the user query.
        4. You have access to the following tools: {tool_descriptions}, always use them to solve the task.
        5. Your responsablity start with a task and finish with an email with a report.
        6. Is recommended to use an exploratory search first, and then with that information, refine your search.

    """)
    return instructions


class ResearchManager:
    def __init__(self):
        tools = [
            web_search_agent.as_tool(tool_name="web_search", tool_description="Perform a web search"),
            writer_agent.as_tool(tool_name="write_report", tool_description="Write a detailed report based on the search results"),
            email_agent.as_tool(tool_name="send_email", tool_description="Send an email with the report"),
        ]
        prompts = create_prompts(tools)
        self.agent = Agent(
            name="ResearchManagerAgent",
            instructions=prompts,
            model="gpt-4o-mini",
            tools=tools,
            output_type=ReportData,
            model_settings=ModelSettings(tool_choice="required", temperature=0.0),
        )

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting research...")
            yield "Searches complete, writing report..."
            result = Runner.run_streamed(
                self.agent,
                f"Query: {query}",
                max_turns = 20,
            )
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    continue
                elif event.type == "agent_updated_stream_event":
                    continue
                # When items are generated, print them
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        # print(f"Tool call: {event.item}")
                        function_name = event.item.raw_item.name
                        if function_name == "plan_searches":
                            yield f"Planning searches..."
                        elif function_name == "search":
                            yield f"Searching for information..."
                        elif function_name == "write_report":
                            yield f"Writing report..."
                    elif event.item.type == "tool_call_output_item":
                        # print(f"Tool call output: {event.item.output}")
                        continue
                    elif event.item.type == "message_output_item":
                        output = ItemHelpers.text_message_output(event.item)
                        parse_output = ReportData.model_validate_json(output)
                        yield f"Final output: {parse_output.markdown_report}"
                    else:
                        pass  # Ignore other event types


#     async def plan_searches(self, query: str) -> WebSearchPlan:
#         """ Plan the searches to perform for the query """
#         print("Planning searches...")
#         result = await Runner.run(
#             planner_agent,
#             f"Query: {query}",
#         )
#         print(f"Will perform {len(result.final_output.searches)} searches")
#         return result.final_output_as(WebSearchPlan)

#     async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
#         """ Perform the searches to perform for the query """
#         print("Searching...")
#         num_completed = 0
#         tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
#         results = []
#         for task in asyncio.as_completed(tasks):
#             result = await task
#             if result is not None:
#                 results.append(result)
#             num_completed += 1
#             print(f"Searching... {num_completed}/{len(tasks)} completed")
#         print("Finished searching")
#         return results

#     async def search(self, item: WebSearchItem) -> str | None:
#         """ Perform a search for the query """
#         input = f"Search term: {item.query}\nReason for searching: {item.reason}"
#         try:
#             result = await Runner.run(
#                 search_agent,
#                 input,
#             )
#             return str(result.final_output)
#         except Exception:
#             return None

#     async def write_report(self, query: str, search_results: list[str]) -> ReportData:
#         """ Write the report for the query """
#         print("Thinking about report...")
#         input = f"Original query: {query}\nSummarized search results: {search_results}"
#         result = await Runner.run(
#             writer_agent,
#             input,
#         )

#         print("Finished writing report")
#         return result.final_output_as(ReportData)
    
#     async def send_email(self, report: ReportData) -> None:
#         print("Writing email...")
#         result = await Runner.run(
#             email_agent,
#             report.markdown_report,
#         )
#         print("Email sent")
#         return report

# # Define the ResearchManager as a meta-agent using other agents as tools
# RESEARCH_MANAGER_INSTRUCTIONS = """
# You are a Research Manager. When given a research query, you have access to four tools:
# 1) planner ➜ returns a WebSearchPlan
# 2) search  ➜ returns raw search results for each WebSearchItem
# 3) writer  ➜ turns the query & summarized search results into a markdown report
# 4) email   ➜ sends the markdown report

# Execute these steps in order and return the final markdown report as your output.
# """
# research_manager_agent = Agent(
#     name="ResearchManagerAgent",
#     instructions=RESEARCH_MANAGER_INSTRUCTIONS,
#     model="gpt-4o-mini",
#     tools={
#         "planner": planner_agent,
#         "search": web_search_agent,
#         "writer": write_reportr_agent,
#         "email": email_agent,
#     },
#     output_type=ReportData,
# )
from agents import Runner, trace, gen_trace_id, Agent, ItemHelpers
from search_agent import search_agent
from planner_agent import planner_agent, WebSearchItem, WebSearchPlan
from writer_agent import writer_agent, ReportData
from email_agent import email_agent
from openai.types.responses import ResponseTextDeltaEvent
import asyncio

class ResearchManager:
    def __init__(self):
        tools = [
            planner_agent.as_tool(tool_name="plan_searches", tool_description="Plan a set of web searches to perform"),
            search_agent.as_tool(tool_name="search", tool_description="Search the web for information"),
            writer_agent.as_tool(tool_name="write_report", tool_description="Write a detailed report based on the search results"),
            email_agent.as_tool(tool_name="send_email", tool_description="Send an email with the report"),
        ]   
        self.agent = Agent(
            name="ResearchManagerAgent",
            instructions=RESEARCH_MANAGER_INSTRUCTIONS,
            model="gpt-4o-mini",
            tools=tools,
            output_type=ReportData,
        )

    async def run(self, query: str):
        """ Run the deep research process, yielding the status updates and the final report"""
        trace_id = gen_trace_id()
        with trace("Research trace", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}")
            # yield f"View trace: https://platform.openai.com/traces/trace?trace_id={trace_id}"
            print("Starting research...")
            result = await Runner.run(
                self.agent,
                f"Query: {query}",
            )
            return result.final_output_as(ReportData)
    async def run1(self, query: str):
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
            )
            async for event in result.stream_events():
                # We'll ignore the raw responses event deltas
                if event.type == "raw_response_event":
                    continue
                # When the agent updates, print that
                elif event.type == "agent_updated_stream_event":
                    yield f"Agent updated: {event.new_agent.name}"
                    continue
                # When items are generated, print them
                elif event.type == "run_item_stream_event":
                    if event.item.type == "tool_call_item":
                        yield "-- Tool was called"
                    elif event.item.type == "tool_call_output_item":
                        yield f"-- Tool output: {event.item.output}"
                    elif event.item.type == "message_output_item":
                        yield f"-- Message output:\n {ItemHelpers.text_message_output(event.item)}"
                        output = ItemHelpers.text_message_output(event.item)
                        parse_output = ReportData.model_validate_json(output)
                        yield f"Final output: {parse_output.markdown_report}"
                    else:
                        pass  # Ignore other event types


    async def plan_searches(self, query: str) -> WebSearchPlan:
        """ Plan the searches to perform for the query """
        print("Planning searches...")
        result = await Runner.run(
            planner_agent,
            f"Query: {query}",
        )
        print(f"Will perform {len(result.final_output.searches)} searches")
        return result.final_output_as(WebSearchPlan)

    async def perform_searches(self, search_plan: WebSearchPlan) -> list[str]:
        """ Perform the searches to perform for the query """
        print("Searching...")
        num_completed = 0
        tasks = [asyncio.create_task(self.search(item)) for item in search_plan.searches]
        results = []
        for task in asyncio.as_completed(tasks):
            result = await task
            if result is not None:
                results.append(result)
            num_completed += 1
            print(f"Searching... {num_completed}/{len(tasks)} completed")
        print("Finished searching")
        return results

    async def search(self, item: WebSearchItem) -> str | None:
        """ Perform a search for the query """
        input = f"Search term: {item.query}\nReason for searching: {item.reason}"
        try:
            result = await Runner.run(
                search_agent,
                input,
            )
            return str(result.final_output)
        except Exception:
            return None

    async def write_report(self, query: str, search_results: list[str]) -> ReportData:
        """ Write the report for the query """
        print("Thinking about report...")
        input = f"Original query: {query}\nSummarized search results: {search_results}"
        result = await Runner.run(
            writer_agent,
            input,
        )

        print("Finished writing report")
        return result.final_output_as(ReportData)
    
    async def send_email(self, report: ReportData) -> None:
        print("Writing email...")
        result = await Runner.run(
            email_agent,
            report.markdown_report,
        )
        print("Email sent")
        return report

# Define the ResearchManager as a meta-agent using other agents as tools
RESEARCH_MANAGER_INSTRUCTIONS = """
You are a Research Manager. When given a research query, you have access to four tools:
1) planner ➜ returns a WebSearchPlan
2) search  ➜ returns raw search results for each WebSearchItem
3) writer  ➜ turns the query & summarized search results into a markdown report
4) email   ➜ sends the markdown report

Execute these steps in order and return the final markdown report as your output.
"""
research_manager_agent = Agent(
    name="ResearchManagerAgent",
    instructions=RESEARCH_MANAGER_INSTRUCTIONS,
    model="gpt-4o-mini",
    tools={
        "planner": planner_agent,
        "search": search_agent,
        "writer": writer_agent,
        "email": email_agent,
    },
    output_type=ReportData,
)
from pydantic import BaseModel, Field
from agents import Agent, output_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents import function_tool
from model import gemini_model
from input_guardrails import cyber_agent_input_guardrail
import re


class ReportData(BaseModel):
    short_summary: str = Field(description="A short summary of the report.")

    markdown_report: str = Field(description="The full markdown report.")

    follow_up_questions: list[str] = Field(
        description="A list of follow-up questions for the user."
    )

template_placeholder_agent = Agent(
    name="Template Placeholder Guardrail check",
    instructions="Check if the input contains template placeholders.",
)



INSTRUCTIONS = (
    "You are a senior researcher tasked with writing a cohesive report for a research query. "
    "You will be provided with the original query, and some initial research done by a research assistant.\n"
    "You should first come up with an outline for the report that describes the structure and "
    "flow of the report. Then, generate the report and return that as your final output.\n"
    "The final output should be in markdown format, and it should be lengthy and detailed. Aim "
    "for 5-10 pages of content, at least 1000 words."
)



writer_agent = Agent(
    name="WriterAgent",
    instructions=INSTRUCTIONS,
    model=gemini_model,
    output_type=ReportData,
    # output_guardrails=[
    #     template_placeholder_agent,
    # ],
    # input_guardrails=[cyber_agent_input_guardrail],
)
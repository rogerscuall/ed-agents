from agents import Agent, handoff, RunContextWrapper
from email_agent import email_agent
from agents.extensions import handoff_filters
from writer_agent import ReportData

async def email_handoff(ctx: RunContextWrapper[None], input_data: ReportData):
    yield f"Handing off to email agent..."

email_handoff_agent = handoff(
    agent=email_agent,
    on_handoff=email_handoff,
    tool_name_override="email_agent",
    tool_description_override="Send an email with a report.",
    input_type=ReportData,
    
)
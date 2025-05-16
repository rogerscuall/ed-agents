from agents import Agent, handoff, RunContextWrapper
from email_agent import email_agent
from agents.extensions import handoff_filters

def email_handoff(ctx: RunContextWrapper[None]):
    yield f"Handing off to email agent..."

email_handoff_agent = handoff(
    agent=email_agent,
    on_handoff=email_handoff,
    tool_name_override="email_agent",
    tool_description="Send an email with a report.",
    input_filter=handoff_filters.remove_all_tools,
)
from email_agent import email_agent
from agents.extensions import handoff_filters
from agents import Agent, function_tool, input_guardrail, Runner, RunContextWrapper, GuardrailFunctionOutput, TResponseInputItem
from agents.extensions.visualization import draw_graph
from agents.model_settings import ModelSettings
from handoff_agents import email_handoff_agent
import asyncio

async def test_email_agent_handoff():

    simple_agent = Agent(
        name="Simple Agent",
        instructions="Use your tools or agent to send an email.",
        output_type=str,
        handoffs=[
            email_handoff_agent,
        ]
    )

    # Create a simple input
    input_data = "Send a reminder of the meeting tomorrow at 10 AM to all team members. Dont add any more information." 

    # Run the agent
    result = await Runner.run(simple_agent, input_data)
    print("Result:", result)

async def test_email_agent():

    simple_agent = email_agent
    # Create a simple input
    input_data = "Send a reminder of the meeting tomorrow at 10 AM to all team members. Include this text: [Your Name]"

    # Run the agent
    result = await Runner.run(simple_agent, input_data)
    print("Result:", result)

if __name__ == "__main__":
    # asyncio.run(test_email_agent())
    asyncio.run(test_email_agent_handoff())
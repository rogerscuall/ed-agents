import os
from typing import Dict
from model import gemini_model
from pydantic import BaseModel, Field
import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, function_tool, output_guardrail, GuardrailFunctionOutput, RunContextWrapper, Runner, OutputGuardrailTripwireTriggered
from agents.extensions.visualization import draw_graph
import re
from textwrap import dedent

class EmailOutput(BaseModel):
    message: str = Field(description="The email message to be sent.")

class PlaceHolderOutput(BaseModel):
    message: str = Field(description="The email message to be sent.")
    tripwire_triggered: bool = Field(description="Whether the tripwire was triggered.")
    reasoning: str = Field(description="The reasoning for the tripwire.")

template_placeholder_agent = Agent(
    name = "Guardrail Place holder agent",
    instructions=dedent(
    """
    Your task is to review the email and verify that there are not placeholders or names.
    Some examples of placeholders are as follow:
    - [Your Name]
    - [Recipient's Name],
    If you find any place holder please set tripwire_triggered.

    """
    ),
    output_type = PlaceHolderOutput
)

@output_guardrail
async def template_placeholder_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: EmailOutput,
    ) -> GuardrailFunctionOutput:
        result = await Runner.run(template_placeholder_agent, output.message, context=ctx.context)
        return GuardrailFunctionOutput(
            output_info=result.final_output_as(PlaceHolderOutput).reasoning,
            tripwire_triggered=result.final_output_as(PlaceHolderOutput).tripwire_triggered,
        )

@function_tool
def send_email(subject: str, html_body: str) -> Dict[str, str]:
    """ Send an email with the given subject and HTML body """
    sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
    from_email = Email("website.stray_1y@icloud.com") # put your verified sender here
    to_email = To("rogerscuall@gmail.com")
    content = Content("text/html", html_body)
    mail = Mail(from_email, to_email, subject, content).get()
    response = sg.client.mail.send.post(request_body=mail)
    print("Email response", response.status_code)
    return {"status": "success"}

INSTRUCTIONS = """You are able to send a nicely formatted HTML email based on a detailed report.
You will be provided with a detailed report. You should use your tool to send one email, providing the 
report converted into clean, well presented HTML with an appropriate subject line."""

email_agent = Agent(
    name="EmailAgent",
    instructions=INSTRUCTIONS,
    # tools=[send_email],
    # model=gemini_model,
    output_type=EmailOutput,
    output_guardrails=[template_placeholder_guardrail],
)

from writer_agent import ReportData

@function_tool
async def send_email_tool(report: ReportData):
   try:
    await Runner.run(email_agent, report.markdown_report)
   except OutputGuardrailTripwireTriggered as e:
    print(f"Guardrail tripwire triggered because: {e.guardrail_result.output.output_info.reasoning}")

EXPERT_INSTRUCTIONS = """
You can read a detailed report and send an email based on it.
You will be provided with a detailed report.
You should use your tool to send one email.
"""

email_expert = Agent(
    name="EmailExpert",
    instructions=INSTRUCTIONS,
    tools=[send_email_tool],
    model=gemini_model
)

if __name__ == "__main__":
    draw_graph(email_agent, "email_agent.png")
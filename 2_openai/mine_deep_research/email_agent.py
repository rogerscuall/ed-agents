import os
from typing import Dict
from model import gemini_model
from pydantic import BaseModel, Field
import sendgrid
from sendgrid.helpers.mail import Email, Mail, Content, To
from agents import Agent, function_tool, output_guardrail, GuardrailFunctionOutput, RunContextWrapper
from agents.extensions.visualization import draw_graph
import re

class EmailOutput(BaseModel):
    message: str = Field(description="The email message to be sent.")

class PlaceHolderOutput(BaseModel):
    message: str = Field(description="The email message to be sent.")
    tripwire_triggered: bool = Field(description="Whether the tripwire was triggered.")
    reasoning: str = Field(description="The reasoning for the tripwire.")

@output_guardrail
async def template_placeholder_guardrail(
        ctx: RunContextWrapper,
        agent: Agent,
        output: EmailOutput,
    ) -> GuardrailFunctionOutput:
        return GuardrailFunctionOutput(
            output_info="text",
            tripwire_triggered=True,
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
    tools=[send_email],
    # model=gemini_model,
    output_type=EmailOutput,
    output_guardrails=[template_placeholder_guardrail],
)

if __name__ == "__main__":
    draw_graph(email_agent, "email_agent.png")

from agents import Agent, function_tool, input_guardrail, Runner, RunContextWrapper, GuardrailFunctionOutput, TResponseInputItem
from agents.extensions.visualization import draw_graph
from pydantic import BaseModel, Field
import re
from agents import input_guardrail, GuardrailFunctionOutput, RunContextWrapper, Agent, TResponseInputItem
class CyberSecurityOutput(BaseModel):
    is_cyber_security_homework: bool = Field(description="Whether the output contains cyber security information.")
    reasoning: str = Field(description="Reasoning behind the classification.")

cyber_agent_output_agent = Agent(
    name="CyberSecurity Guardrail check",
    instructions="Check if the output has cyber security information.",
    output_type=CyberSecurityOutput,
)

@input_guardrail
async def cyber_agent_input_guardrail(ctx: RunContextWrapper[None], agent: Agent, input: str | list[TResponseInputItem]) -> GuardrailFunctionOutput:
    result = await Runner.run(cyber_agent_output_agent, input, context=ctx.context)
    return GuardrailFunctionOutput(output_info=result.final_output,tripwire_triggered=result.final_output.is_cyber_security_homework,)


template_placeholder_agent = Agent(
    name="Template Placeholder Guardrail check",
    instructions="Check if the input contains template placeholders.",
)

@input_guardrail
async def template_placeholder_guardrail(
        ctx: RunContextWrapper[None],
        agent: Agent,
        input: str | list[TResponseInputItem]
    ) -> GuardrailFunctionOutput:
        # Normalize input to a single string
        text = (
            input
            if isinstance(input, str)
            else " ".join(item.content for item in input)
        )

        # Find any bracketed placeholders like [Something]
        placeholders = re.findall(r"\[[^\]]+\]", text)
        tripwire = bool(placeholders)

        return GuardrailFunctionOutput(
            output_info=input,
            tripwire_triggered=tripwire,
        )
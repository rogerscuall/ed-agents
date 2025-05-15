import gradio as gr
from dotenv import load_dotenv
from research_manager import ResearchManager
from agents import Runner, trace
import asyncio

load_dotenv(override=True)
final_agent = ResearchManager()

async def main(query: str):
    async for chunk in final_agent.run(query):
        print(chunk)

if __name__ == "__main__":
    asyncio.run(main("Latest latest updates of the IRS 2024 tax code changes"))

# async def run(query: str):    
#     async for chunk in Runner.stream(research_manager_agent, query):
#         yield chunk


# with gr.Blocks(theme=gr.themes.Default(primary_hue="sky")) as ui:
#     gr.Markdown("# Deep Research")
#     query_textbox = gr.Textbox(label="What topic would you like to research?")
#     run_button = gr.Button("Run", variant="primary")
#     report = gr.Markdown(label="Report")
    
#     run_button.click(fn=run, inputs=query_textbox, outputs=report)
#     query_textbox.submit(fn=run, inputs=query_textbox, outputs=report)

# ui.launch(inbrowser=True)


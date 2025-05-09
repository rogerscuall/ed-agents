from phoenix.otel import register
import os
from dotenv import load_dotenv
# PHOENIX_API_KEY = "d9d7c75a07497007829:0c18f52"
# os.environ["PHOENIX_CLIENT_HEADERS"] = f"api_key={PHOENIX_API_KEY}"
# os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"
# configure the Phoenix tracer
tracer_provider = register(
  project_name="agents", # Default is 'default'
  auto_instrument=True # Auto-instrument your app based on installed dependencies
) 
load_dotenv(override=True)

from agents import Agent, Runner

agent = Agent(name="agent", instructions="You are a helpful assistant")
result = Runner.run_sync(agent, "Write a haiku about recursion in programming.")
print(result.final_output)
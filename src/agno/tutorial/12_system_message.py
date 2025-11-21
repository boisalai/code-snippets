"""
See https://docs.agno.com/concepts/agents/context
"""
# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.huggingface import HuggingFace

model_id1 = "meta-llama/Llama-3.3-70B-Instruct"
model_id2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id3 = "Qwen/Qwen2.5-7B-Instruct"
model_id4 = "mistralai/Mistral-7B-Instruct-v0.2"

hf_model = HuggingFace(model_id4)

agent = Agent(
    model=hf_model,
    description="You are a famous short story writer asked to write for a magazine",
    instructions=["Always write 2 sentence stories."],
    markdown=True,
    debug_mode=True,  # Set to True to view the detailed logs and see the compiled system message
)
agent.print_response("Tell me a horror story.", stream=True)
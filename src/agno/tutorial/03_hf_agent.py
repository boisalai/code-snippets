"""
See https://docs.agno.com/concepts/models/huggingface
"""

# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.huggingface import HuggingFace

model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "Qwen/Qwen2.5-7B-Instruct"

agent = Agent(
    model=HuggingFace(
        id=model_id,
        max_tokens=4096,
    ),
    markdown=True,
)

response = agent.run(
    "Write a creative short story (maximum 2 sentences) about a robot. "
    "Respond directly with the story text only."
)

print(response.content)
print(response.run_id)
print(response.session_id)
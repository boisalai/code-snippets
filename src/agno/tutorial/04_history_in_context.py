"""
https://docs.agno.com/concepts/agents/sessions#history-in-context
"""

# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.db.in_memory import InMemoryDb

model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "Qwen/Qwen2.5-7B-Instruct"

agent = Agent(
    model=HuggingFace(
        id=model_id,
        max_tokens=4096,
    ),
    db=InMemoryDb()
)

agent.print_response("Hi, I'm John. Nice to meet you!")

agent.print_response("What is my name?", add_history_to_context=True)
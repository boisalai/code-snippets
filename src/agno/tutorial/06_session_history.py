"""
https://docs.agno.com/concepts/agents/sessions#history-in-context
"""

# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.db.sqlite import SqliteDb

model_id1 = "meta-llama/Llama-3.3-70B-Instruct"
model_id2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id3 = "Qwen/Qwen2.5-7B-Instruct"

agent = Agent(
    model=HuggingFace(id=model_id3),
    db=SqliteDb(db_file="tmp/data.db"),
    add_history_to_context=True,
    num_history_runs=3,
    read_chat_history=True,
    description="You are a helpful assistant that always responds in a polite, upbeat and positive manner.",
)

agent.print_response("Share a 2 sentence horror story", stream=True)

agent.print_response("What was my first message?", stream=True)

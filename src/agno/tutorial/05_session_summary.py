"""
https://docs.agno.com/concepts/agents/sessions#history-in-context
"""

# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.db.sqlite import SqliteDb

db = SqliteDb(db_file="tmp/data.db")

user_id = "jon_hamm@example.com"
session_id = "1001"

model_id = "meta-llama/Llama-3.3-70B-Instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id = "Qwen/Qwen2.5-7B-Instruct"

agent = Agent(
    model=HuggingFace(id=model_id),
    db=db,
    enable_session_summaries=True,
)

agent.print_response(
    "What can you tell me about quantum computing?",
    stream=True,
    user_id=user_id,
    session_id=session_id,
)

agent.print_response(
    "I would also like to know about LLMs?",
    stream=True,
    user_id=user_id,
    session_id=session_id
)

session_summary = agent.get_session_summary(session_id=session_id)
print(f"Session summary: {session_summary.summary}")
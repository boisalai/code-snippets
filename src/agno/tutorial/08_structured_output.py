"""
https://docs.agno.com/concepts/agents/input-output
"""
from typing import List
from pydantic import BaseModel, Field
from agno.agent import Agent
from agno.models.huggingface import HuggingFace

from dotenv import load_dotenv
load_dotenv()

class MovieScript(BaseModel):
    setting: str = Field(..., description="Provide a nice setting for a blockbuster movie.")
    ending: str = Field(..., description="Ending of the movie. If not available, provide a happy ending.")
    genre: str = Field(
        ..., description="Genre of the movie. If not available, select action, thriller or romantic comedy."
    )
    name: str = Field(..., description="Give a name to this movie")
    characters: List[str] = Field(..., description="Name of characters for this movie.")
    storyline: str = Field(..., description="3 sentence storyline for the movie. Make it exciting!")

# Agent that uses structured outputs
structured_output_agent = Agent(
    # model=OpenAIChat(id="gpt-5-mini"),
    # model=Ollama(id="qwen3"),
    model=HuggingFace(id="meta-llama/Meta-Llama-3-8B-Instruct"),
    description="You write movie scripts.",
    output_schema=MovieScript,
    use_json_mode=True,
)

structured_output_agent.print_response("New York")



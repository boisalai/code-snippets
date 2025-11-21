"""
See https://docs.agno.com/concepts/agents/input-output
"""
# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

from typing import List

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.tools.hackernews import HackerNewsTools
from pydantic import BaseModel, Field
from rich.pretty import pprint


# Define your input schema
class ResearchTopic(BaseModel):
    topic: str
    sources_required: int = Field(description="Number of sources", default=5)


# Define your output schema
class ResearchOutput(BaseModel):
    summary: str = Field(..., description="Executive summary of the research")
    insights: List[str] = Field(..., description="Key insights from posts")
    top_stories: List[str] = Field(
        ..., description="Most relevant and popular stories found"
    )
    technologies: List[str] = Field(
        ..., description="Technologies mentioned"
    )
    sources: List[str] = Field(..., description="Links to the most relevant posts")


model_id1 = "meta-llama/Llama-3.3-70B-Instruct"
model_id2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id3 = "Qwen/Qwen2.5-7B-Instruct"
model_id4 = "mistralai/Mistral-7B-Instruct-v0.2"

# Les modèles HuggingFace open-source sont généralement moins bons que Claude 
# pour les sorties structurées complexes. 
# Ce code fonctionne avec Qwen2.5-7B-Instruct.
hf_model = HuggingFace(model_id3)

# Define your agent
hn_researcher_agent = Agent(
    # Model to use - HuggingFace fait le travail principal
    model=hf_model,
    # Tools to use
    tools=[HackerNewsTools()],
    instructions=[
        "You are a research agent that analyzes HackerNews posts.",
        "Use the get_top_hackernews_stories tool to fetch relevant stories.",
        "Analyze the stories and extract key information.",
        "Provide a comprehensive analysis including summary, insights, top stories, technologies, and sources.",
    ],
    # Add your input schema
    input_schema=ResearchTopic,
    # Add your output schema
    output_schema=ResearchOutput,
    # Claude parse la sortie de HuggingFace pour garantir le format structuré
    parser_model=hf_model,
    # Désactiver structured_outputs pour HF (il n'est pas aussi bon que Claude pour ça)
    structured_outputs=False,
)

# Run the Agent
response = hn_researcher_agent.run(
    input=ResearchTopic(topic="AI", sources_required=5)
)

# Print the response
pprint(response.content)
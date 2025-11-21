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


class ResearchTopic(BaseModel):
    """Structured research topic with specific requirements"""

    topic: str
    focus_areas: List[str] = Field(description="Specific areas to focus on")
    target_audience: str = Field(description="Who this research is for")
    sources_required: int = Field(description="Number of sources needed", default=5)

hackernews_agent = Agent(
    name="Hackernews Agent",
    model=HuggingFace(id="meta-llama/Llama-3.2-3B-Instruct"),
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
    markdown=True,
)

hackernews_agent.print_response(
    input=ResearchTopic(
        topic="AI",
        focus_areas=["AI", "Machine Learning"],
        target_audience="Developers",
        sources_required=5,
    )
)
"""
https://docs.agno.com/concepts/workflows/running-workflow
"""
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.db.sqlite import SqliteDb
from agno.team import Team
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.tools.hackernews import HackerNewsTools
from agno.workflow import Step, Workflow
from agno.run.workflow import WorkflowRunOutput
from agno.utils.pprint import pprint_run_response


model_id1 = "meta-llama/Llama-3.3-70B-Instruct"
model_id2 = "meta-llama/Meta-Llama-3-8B-Instruct"
model_id3 = "Qwen/Qwen2.5-7B-Instruct"
model_id4 = "mistralai/Mistral-7B-Instruct-v0.2"

hf_model = HuggingFace(model_id4)


# Define agents
hackernews_agent = Agent(
    name="Hackernews Agent",
    model=hf_model,
    tools=[HackerNewsTools()],
    role="Extract key insights and content from Hackernews posts",
)
web_agent = Agent(
    name="Web Agent",
    model=hf_model,
    tools=[DuckDuckGoTools()],
    role="Search the web for the latest news and trends",
)

# Define research team for complex analysis
research_team = Team(
    name="Research Team",
    model=hf_model,
    members=[hackernews_agent, web_agent],
    instructions="Research tech topics from Hackernews and the web",
)

content_planner = Agent(
    name="Content Planner",
    model=hf_model,
    instructions=[
        "Plan a content schedule over 4 weeks for the provided topic and research content",
        "Ensure that I have posts for 3 posts per week",
    ],
)

content_creation_workflow = Workflow(
    name="Content Creation Workflow",
    description="Automated content creation from blog posts to social media",
    db=SqliteDb(db_file="tmp/workflow.db"),
    steps=[research_team, content_planner],
)

# Create and use workflow
if __name__ == "__main__":
    response: WorkflowRunOutput = content_creation_workflow.run(
        input="AI trends in 2024",
        markdown=True,
    )

    pprint_run_response(response, markdown=True)
"""
See https://docs.agno.com/introduction/quickstart
See https://docs.agno.com/concepts/agents/building-agents
"""

from urllib import response
from dotenv import load_dotenv
load_dotenv()

from agno.agent import Agent, RunOutput, RunEvent
from agno.models.ollama import Ollama
from agno.models.anthropic import Claude
from agno.models.huggingface import HuggingFace
from agno.tools.hackernews import HackerNewsTools
from agno.utils.pprint import pprint_run_response

def claude_agent():
    agent = Agent(
        model=Claude(id="claude-sonnet-4-5"),
        tools=[HackerNewsTools()],
        instructions="Write a report on the topic. Output only the report.",
        markdown=True,
    )
    agent.print_response("Trending startups and products.", stream=True)

def ollama_agent():
    agent = Agent(
        model=Ollama(id="qwen3"),
        tools=[HackerNewsTools()],
        instructions="Write a report on the topic. Output only the report.",
        markdown=True,
    )
    # agent.print_response("Trending startups and products.", stream=True)

    # Run agent and return the response as a variable
    response: RunOutput = agent.run("Trending startups and products.")
    # Print the response
    print(response.content)
    # Print the response in markdown format
    # pprint_run_response(response, markdown=True)

def ollama_agent_stream():
    # https://docs.agno.com/concepts/agents/running-agents#handling-events
    agent = Agent(
        model=Ollama(id="qwen3"),
        tools=[HackerNewsTools()],
        instructions="Write a report on the topic. Output only the report.",
        markdown=True,
    )

    stream = agent.run("Trending products", stream=True, stream_events=True)

    for chunk in stream:
        if chunk.event == RunEvent.run_content:
            print(f"Content: {chunk.content}")
        elif chunk.event == RunEvent.tool_call_started:
            print(f"Tool call started: {chunk.tool.tool_name}")
        elif chunk.event == RunEvent.reasoning_step:
            print(f"Reasoning step: {chunk.content}")

def hf_agent():
    model_id = "Qwen/Qwen2.5-7B-Instruct"

    agent = Agent(
        model=HuggingFace(
            id=model_id,
            max_tokens=4096,
        ),
        tools=[HackerNewsTools()],
        instructions="Write a report on the topic. Output only the report.",
        markdown=True,
    )

    # Run agent and return the response as a variable
    response: RunOutput = agent.run("Trending startups and products.")
    # Print the response
    print(response.content)



if __name__ == "__main__":
    # claude_agent()
    # ollama_agent()
    # ollama_agent_stream()
    hf_agent()
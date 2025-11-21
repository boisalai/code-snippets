"""
See https://docs.agno.com/concepts/models/ollama
"""
from agno.agent import Agent
from agno.models.ollama import Ollama

def ollama_agent():
    agent = Agent(
        model=Ollama(id="llama3.2"),
        markdown=True
    )

    # Print the response in the terminal
    agent.print_response("Share a 2 sentence horror story.")


if __name__ == "__main__":
    ollama_agent()
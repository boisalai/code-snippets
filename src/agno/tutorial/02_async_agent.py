"""
https://docs.agno.com/examples/concepts/agent/async/basic
"""
from dotenv import load_dotenv
load_dotenv()

import asyncio

from agno.agent import Agent
from agno.models.huggingface import HuggingFace
from agno.utils.pprint import apprint_run_response

model_id = "Qwen/Qwen2.5-7B-Instruct"
model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

agent = Agent(
    model=HuggingFace(id=model_id),
)


async def basic():
    response = await agent.arun(input="Tell me a joke.")
    print(response.content)


async def basic_print():
    await agent.aprint_response(input="Tell me a joke.")


async def basic_pprint():
    response = await agent.arun(input="Tell me a joke.")
    await apprint_run_response(response)


if __name__ == "__main__":
    asyncio.run(basic())
    # OR
    # asyncio.run(basic_print())
    # OR
    # asyncio.run(basic_pprint())
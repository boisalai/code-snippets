from agno.agent import Agent
from agno.models.huggingface import HuggingFace

# We need to load HF_TOKEN API key from .env file
from dotenv import load_dotenv
load_dotenv()

# ************ Create Agent with Response Caching ************
agent = Agent(
    model=HuggingFace(
        id="meta-llama/Meta-Llama-3-8B-Instruct",  # Try the 8B version instead
        cache_response=True,      # Enable response caching
        cache_ttl=3600,           # Optional: Cache for 1 hour
        cache_dir="./cache"       # Optional: Custom cache directory
    )
)

# ************ First run – Cache Miss (takes normal time) ************
response = agent.run("Write me a short story about a cat that can talk.")
print(f"First run: {response.metrics.duration:.3f}s")

# ************ Second run – Cache Hit (instant!) ************
response = agent.run("Write me a short story about a cat that can talk.")
print(f"Second run: {response.metrics.duration:.3f}s")  # Much faster!

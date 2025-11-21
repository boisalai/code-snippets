from agno.agent import Agent
from agno.models.openai.like import OpenAILike

# See https://huggingface.co/mlx-community/models?sort=likes
model_id1 = "mlx-community/Qwen3-8B-8bit"
model_id2 = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

agent = Agent(
    model=OpenAILike(
        id=model_id2,
        api_key="not-needed",
        base_url="http://localhost:8080/v1",
    )
)

agent.print_response("Qu'est-ce que le calcul quantique ?")
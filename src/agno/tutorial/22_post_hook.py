import asyncio
from typing import Any, Dict

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.run.agent import RunOutput

# ************ Define your post-hook ************
def send_notification(run_output: RunOutput, metadata: Dict[str, Any]) -> None:
    """Post-hook: Send a notification after agent completes"""
    email = metadata.get("email")
    if email:
        send_email(email, run_output.content)

def send_email(email: str, content: str) -> None:
    """Send an email to the user (mock for example)"""
    print("\nEMAIL NOTIFICATION")
    print(f"To: {email}")
    print("Subject: Your content is ready!")
    print(f"Preview: {content[:150]}...\n")

model_id1 = "mlx-community/Qwen3-8B-8bit"
model_id2 = "mlx-community/Qwen3-4B-4bit"
model=OpenAILike(
        id=model_id2,
        api_key="not-needed",
        base_url="http://localhost:8080/v1",
    )

# ************ Create Agent with post-hook ************
agent = Agent(
    name="Content Writer",
    model=model,
    post_hooks=[send_notification],
    instructions=[
        "You are a helpful content writer.",
        "Create clear and engaging content.",
        "Keep responses concise and well-structured.",
    ],
)

# ************ Run your agent ************
async def main():
    await agent.aprint_response(
        "Write a brief introduction about the benefits of AI automation.",
        user_id="user_123",
        metadata={"email": "ay.boisvert@gmail.com"},
    )

if __name__ == "__main__":
    asyncio.run(main())

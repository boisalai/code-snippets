from mlx_lm import load, generate
from mlx_lm.sample_utils import make_sampler

def mlx():
    # Using mlx_lm directly (without LangChain wrapper)
    # Load the model and tokenizer
    model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    # Prepare the prompt in chat format
    messages = [
        {"role": "user", "content": "The sky is"}
    ]

    # Convert to format expected by the model
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Create a sampler with temperature
    sampler = make_sampler(temp=0.7)

    # Generate response
    response = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=100,
        sampler=sampler,
        verbose=False
    )

    print(response)

if __name__ == "__main__":
    mlx()
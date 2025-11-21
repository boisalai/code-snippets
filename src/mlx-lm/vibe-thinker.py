"""
See https://huggingface.co/mlx-community/VibeThinker-1.5B-mlx-4bit
"""
from mlx_lm import load, generate

model, tokenizer = load("mlx-community/VibeThinker-1.5B-mlx-4bit")
response = generate(model, tokenizer, prompt="A café sells 6 types of drinks. A customer orders exactly 8 drinks (types may repeat), but no drink type may be chosen more than 3 times. How many distinct combinations of 8 drinks can the customer order? Give your answer as an integer.", verbose=True)

"""
Le 20 novembre 2025, j'obtiens: No text generated for this prompt
"""


"""
Version optimisée avec streaming pour une réponse en temps réel.
Meilleure expérience utilisateur sur MacBook Pro.
"""
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

# Charger le modèle
model, tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

# Préparer le prompt au format chat
messages = [
    {"role": "user", "content": "Explique-moi ce qu'est l'apprentissage automatique en quelques phrases simples."}
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)

# Créer un sampler avec température
sampler = make_sampler(temp=0.7)

# Générer avec streaming (affichage en temps réel)
print("Réponse: ", end="", flush=True)
for response in stream_generate(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt,
    max_tokens=200,
    sampler=sampler
):
    print(response.text, end="", flush=True)
print()  # Nouvelle ligne à la fin

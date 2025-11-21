from dotenv import load_dotenv

# Charger les variables d'environnement depuis le fichier .env
load_dotenv()

import os
print(os.getenv("ANTHROPIC_API_KEY"))
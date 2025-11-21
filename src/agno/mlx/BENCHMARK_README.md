# 🚀 Guide de Benchmark LLM pour MacBook Pro M1

Ce guide vous aide à benchmarker différents modèles LLM (7-8B) sur votre MacBook Pro M1 16GB.

## 📋 Préparation

### 1. Installation d'Ollama (Recommandé - Le plus simple)

```bash
# Installer Ollama
brew install ollama

# Démarrer le service Ollama
ollama serve

# Télécharger les modèles (dans un autre terminal)
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull llama3.2:3b
```

### 2. Configuration MLX (Optionnel - Meilleure performance)

MLX est optimisé pour Apple Silicon et offre généralement les meilleures performances sur M1/M2.

**IMPORTANT:** MLX ne peut exécuter qu'un seul modèle à la fois!

```bash
# Installer mlx-lm
pip install mlx-lm

# Démarrer le serveur MLX avec UN modèle
mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Dans benchmark.py, décommentez et configurez (ligne 89-90):
# MLX_CURRENT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
```

**Pour tester plusieurs modèles MLX:**

1. **Tester le premier modèle:**
   ```bash
   # Terminal 1: Démarrer serveur
   mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

   # Terminal 2: Éditer benchmark.py
   # MLX_CURRENT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

   # Lancer benchmark
   uv run src/agno/mlx/benchmark.py
   ```

2. **Tester le deuxième modèle:**
   ```bash
   # Terminal 1: Ctrl+C puis redémarrer avec nouveau modèle
   mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit

   # Terminal 2: Éditer benchmark.py
   # MLX_CURRENT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"

   # Relancer benchmark
   uv run src/agno/mlx/benchmark.py
   ```

**Modèles MLX recommandés:**
- `mlx-community/Meta-Llama-3-8B-Instruct-4bit` - Excellent équilibre
- `mlx-community/Qwen2.5-7B-Instruct-4bit` - Très bon en multilingue
- `mlx-community/Mistral-7B-Instruct-v0.3-4bit` - Bon raisonnement

### 3. Configuration HuggingFace (Optionnel - Nécessite internet)

```bash
# 1. Créer un compte sur https://huggingface.co
# 2. Générer un token: https://huggingface.co/settings/tokens
# 3. Ajouter le token dans votre fichier .env
echo "HF_TOKEN=hf_your_token_here" >> .env

# 4. Accepter les licences des modèles (si nécessaire)
# Visitez et cliquez "Agree and access repository":
# - https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct
# - https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
# - https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2
```

## 🏃 Exécution du Benchmark

### Scénario 1: Ollama + HuggingFace uniquement (simple)

```bash
# Première exécution (téléchargement)
uv run src/agno/mlx/benchmark.py

# Deuxième exécution (benchmark réel)
uv run src/agno/mlx/benchmark.py

# MLX sera automatiquement skippé (MLX_CURRENT_MODEL = None)
```

### Scénario 2: Ollama + HuggingFace + UN modèle MLX

```bash
# Terminal 1: Démarrer serveur MLX
mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Terminal 2: Éditer benchmark.py (ligne 89)
# MLX_CURRENT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"

# Lancer benchmark
uv run src/agno/mlx/benchmark.py
```

### Scénario 3: Benchmark complet avec TOUS les modèles MLX

Pour tester les 3 modèles MLX, vous devez faire **3 exécutions séparées**:

**Exécution 1 - Llama 3:**
```bash
# Terminal 1
mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit

# Terminal 2: Éditer benchmark.py
# MLX_CURRENT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"
uv run src/agno/mlx/benchmark.py
# Sauvegarder les résultats quelque part
```

**Exécution 2 - Qwen 2.5:**
```bash
# Terminal 1: Ctrl+C puis
mlx_lm.server --model mlx-community/Qwen2.5-7B-Instruct-4bit

# Terminal 2: Éditer benchmark.py
# MLX_CURRENT_MODEL = "mlx-community/Qwen2.5-7B-Instruct-4bit"
uv run src/agno/mlx/benchmark.py
# Sauvegarder les résultats
```

**Exécution 3 - Mistral:**
```bash
# Terminal 1: Ctrl+C puis
mlx_lm.server --model mlx-community/Mistral-7B-Instruct-v0.3-4bit

# Terminal 2: Éditer benchmark.py
# MLX_CURRENT_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
uv run src/agno/mlx/benchmark.py
# Sauvegarder les résultats
```

**Note:** Vous pouvez comparer manuellement les résultats des 3 exécutions pour choisir le meilleur modèle MLX.

## 📊 Types de tests effectués

Le benchmark teste 4 types de tâches:

1. **Q&A Simple** - Questions-réponses de base
2. **Code Generation** - Génération de code Python
3. **Reasoning** - Résolution de problèmes logiques
4. **Summary** - Résumé de texte

## 🎯 Que tester?

### Configuration minimale (Ollama uniquement)
- Temps: ~5-10 minutes
- Modèles: 4 modèles Ollama
- Tests: 16 tests (4 modèles × 4 tâches)

### Configuration complète (Tous les providers)
- Temps: ~30-60 minutes (ou plus avec MLX)
- Modèles: 10 modèles (4 Ollama + 3 MLX + 3 HuggingFace)
- Tests: 40 tests
- **Note MLX:** Nécessite 3 exécutions séparées du script
  - Une exécution par modèle MLX (redémarrer le serveur entre chaque)
  - Ou testez seulement un modèle MLX (configurez MLX_CURRENT_MODEL)

## 📈 Résultats attendus sur M1 16GB

### Vitesse (tokens/seconde)
- **MLX (4-bit)**: ~40-60 tok/s (le plus rapide)
- **Ollama**: ~20-40 tok/s
- **HuggingFace API**: ~10-20 tok/s (variable, dépend de la connexion)

### Qualité (subjective)
- **Llama 3.1 8B**: Excellente qualité générale
- **Qwen 2.5 7B**: Meilleur en multilingue (français)
- **Mistral 7B**: Bon pour le raisonnement

### Usage mémoire
- Modèles 3B: ~4-6 GB RAM
- Modèles 7-8B: ~8-12 GB RAM
- 4-bit quantization (MLX): ~4-6 GB pour 8B

## 🔧 Personnalisation

### Modifier les modèles testés

Éditez `benchmark.py` (lignes 67-87):

```python
# Ajouter/supprimer des modèles Ollama
OLLAMA_MODELS = [
    "llama3.1:8b",
    # Ajoutez vos modèles ici
]

# Ajouter/supprimer des modèles MLX
MLX_MODELS = [
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    # Ajoutez vos modèles ici
]
```

### Modifier les tâches de test

Éditez `benchmark.py` (lignes 38-59):

```python
TEST_CASES = [
    {
        "name": "Ma tâche custom",
        "prompt": "Votre prompt ici",
        "category": "custom"
    },
    # Ajoutez vos tests ici
]
```

## ❓ Troubleshooting

### Ollama: "model not found"
```bash
ollama list  # Vérifier les modèles installés
ollama pull <model-name>  # Télécharger un modèle manquant
```

### MLX: "Connection refused"
```bash
# Le serveur MLX n'est pas démarré
mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
```

### HuggingFace: "API key required"
```bash
# Vérifier que HF_TOKEN est dans .env
cat .env | grep HF_TOKEN
```

### HuggingFace: "Gated model"
- Visitez la page du modèle sur HuggingFace
- Cliquez "Agree and access repository"
- Attendez l'approbation (généralement instantané)

## 💡 Recommandations pour M1 16GB

**Pour la vitesse maximale:**
- Utilisez MLX avec quantization 4-bit
- Modèle recommandé: `mlx-community/Qwen2.5-7B-Instruct-4bit`

**Pour la meilleure qualité:**
- Utilisez Ollama avec Llama 3.1 8B
- Modèle recommandé: `llama3.1:8b`

**Pour le meilleur équilibre:**
- Utilisez Ollama avec Qwen 2.5 7B
- Modèle recommandé: `qwen2.5:7b`

**Pour économiser la RAM:**
- Utilisez des modèles 3B comme `llama3.2:3b`
- Ou utilisez MLX avec 4-bit quantization

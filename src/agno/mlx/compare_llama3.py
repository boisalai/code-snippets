"""
Script simple pour comparer deux versions de Llama 3 8B:
- MLX 4-bit (optimisé Apple Silicon, local)
- HuggingFace (via API, non quantifié)
"""
from dotenv import load_dotenv
load_dotenv()

import time
from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.models.huggingface import HuggingFace


# Quelques prompts de test simples
TEST_PROMPTS = [
    "Qu'est-ce que Python ? Réponds en 2 phrases.",
    "Écris une fonction qui additionne deux nombres.",
    "Explique la différence entre une liste et un tuple en Python.",
]


def test_model(agent: Agent, model_name: str, prompts: list) -> dict:
    """Teste un modèle avec plusieurs prompts"""
    print(f"\n📊 Test de {model_name}")
    print("-" * 60)

    results = []

    for i, prompt in enumerate(prompts, 1):
        print(f"\n{i}. Prompt: {prompt[:50]}...")

        try:
            start = time.time()
            response = agent.run(prompt, stream=False)
            duration = time.time() - start

            # Estimation des tokens (approximatif: ~4 chars par token)
            response_text = response.content if hasattr(response, 'content') else str(response)
            tokens = len(response_text) / 4
            tok_per_sec = tokens / duration if duration > 0 else 0

            print(f"   ✓ {duration:.2f}s | {tok_per_sec:.1f} tok/s | {len(response_text)} chars")
            print(f"   Réponse: {response_text[:100]}...")

            results.append({
                'success': True,
                'time': duration,
                'tok_per_sec': tok_per_sec,
                'length': len(response_text)
            })

        except Exception as e:
            print(f"   ✗ Erreur: {str(e)[:60]}")
            results.append({'success': False, 'error': str(e)})

    return results


def print_comparison(mlx_results: list, hf_results: list):
    """Affiche une comparaison des deux modèles"""
    print("\n" + "=" * 60)
    print("📈 COMPARAISON")
    print("=" * 60)

    # Filtrer les résultats réussis
    mlx_ok = [r for r in mlx_results if r.get('success')]
    hf_ok = [r for r in hf_results if r.get('success')]

    if not mlx_ok or not hf_ok:
        print("⚠️  Impossible de comparer - certains tests ont échoué")
        return

    # Moyennes
    mlx_avg_time = sum(r['time'] for r in mlx_ok) / len(mlx_ok)
    mlx_avg_tok = sum(r['tok_per_sec'] for r in mlx_ok) / len(mlx_ok)

    hf_avg_time = sum(r['time'] for r in hf_ok) / len(hf_ok)
    hf_avg_tok = sum(r['tok_per_sec'] for r in hf_ok) / len(hf_ok)

    print(f"\n🍎 MLX 4-bit (Local):")
    print(f"   Temps moyen: {mlx_avg_time:.2f}s")
    print(f"   Vitesse moyenne: {mlx_avg_tok:.1f} tok/s")

    print(f"\n🤗 HuggingFace (API):")
    print(f"   Temps moyen: {hf_avg_time:.2f}s")
    print(f"   Vitesse moyenne: {hf_avg_tok:.1f} tok/s")

    print(f"\n🏆 Gagnant:")
    if mlx_avg_tok > hf_avg_tok:
        speedup = mlx_avg_tok / hf_avg_tok
        print(f"   MLX est {speedup:.1f}x plus rapide que HuggingFace")
    else:
        speedup = hf_avg_tok / mlx_avg_tok
        print(f"   HuggingFace est {speedup:.1f}x plus rapide que MLX")


def main():
    print("🚀 Comparaison Llama 3 8B: MLX vs HuggingFace")
    print("=" * 60)

    # Configuration MLX (serveur local)
    print("\n1️⃣  Configuration MLX:")
    print("   Modèle: mlx-community/Meta-Llama-3-8B-Instruct-4bit")
    print("   Assurez-vous que le serveur MLX tourne:")
    print("   mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit")

    # Configuration HuggingFace
    print("\n2️⃣  Configuration HuggingFace:")
    print("   Modèle: meta-llama/Meta-Llama-3-8B-Instruct")
    print("   Nécessite: HF_TOKEN dans .env")

    input("\n⏸️  Appuyez sur Entrée pour commencer...")

    # Test MLX
    try:
        mlx_agent = Agent(
            model=OpenAILike(
                id="mlx-community/Meta-Llama-3-8B-Instruct-4bit",
                api_key="not-needed",
                base_url="http://localhost:8080/v1"
            ),
            markdown=False
        )
        mlx_results = test_model(mlx_agent, "MLX 4-bit", TEST_PROMPTS)
    except Exception as e:
        print(f"\n❌ Erreur MLX: {e}")
        print("   Vérifiez que le serveur MLX tourne sur le port 8080")
        mlx_results = []

    # Test HuggingFace
    try:
        hf_agent = Agent(
            model=HuggingFace(
                id="meta-llama/Meta-Llama-3-8B-Instruct",
                max_tokens=512
            ),
            markdown=False
        )
        hf_results = test_model(hf_agent, "HuggingFace", TEST_PROMPTS)
    except Exception as e:
        print(f"\n❌ Erreur HuggingFace: {e}")
        print("   Vérifiez que HF_TOKEN est configuré dans .env")
        hf_results = []

    # Comparaison
    if mlx_results and hf_results:
        print_comparison(mlx_results, hf_results)

    print("\n✅ Terminé!")


if __name__ == "__main__":
    main()

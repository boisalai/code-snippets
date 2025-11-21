"""
Benchmark script for testing LLM performance on MacBook Pro M1 16GB
Tests Ollama (local), MLX (Apple Silicon optimized), and HuggingFace models
Focus: 7-8B models, quality-first, mixed task types
"""
from dotenv import load_dotenv
load_dotenv()

import time
import psutil
from typing import Dict, List, Any
from dataclasses import dataclass
from tabulate import tabulate

from agno.agent import Agent
from agno.models.openai.like import OpenAILike
from agno.models.huggingface import HuggingFace
from agno.models.ollama import Ollama


@dataclass
class BenchmarkResult:
    """Store benchmark results for each model/task combination"""
    model_name: str
    provider: str
    task_name: str
    response_time: float
    tokens_per_second: float
    response_length: int
    success: bool
    error: str = ""
    memory_used_mb: float = 0


# ============================================================
# TEST CASES - Mix of different task types
# ============================================================
TEST_CASES = [
    {
        "name": "Q&A Simple",
        "prompt": "Qu'est-ce que le calcul quantique ? Réponds en 2-3 phrases.",
        "category": "text_generation"
    },
    {
        "name": "Code Generation",
        "prompt": "Écris une fonction Python qui calcule la suite de Fibonacci de manière récursive avec mémoization.",
        "category": "code"
    },
    {
        "name": "Reasoning",
        "prompt": "Si un train part de Paris à 14h et roule à 200 km/h, et un autre part de Lyon (450 km) à 14h30 à 180 km/h vers Paris, à quelle heure se croisent-ils?",
        "category": "reasoning"
    },
    {
        "name": "Summary",
        "prompt": "Résume en 3 points clés: L'intelligence artificielle transforme notre société. Les modèles de langage permettent d'automatiser des tâches complexes. Cependant, ils posent des questions éthiques importantes sur la vie privée et l'emploi.",
        "category": "summary"
    }
]


# ============================================================
# MODEL CONFIGURATIONS
# ============================================================

# Ollama models (optimized for Apple Silicon)
OLLAMA_MODELS = [
    "llama3.2:3b",      # Fast, good quality
    "llama3.1:8b",      # Best quality for 8B
    "qwen2.5:7b",       # Excellent multilingual
    "mistral:7b",       # Good reasoning
]

# MLX models (requires mlx-lm server running on port 8080)
# IMPORTANT: MLX can only run ONE model at a time!
# Set MLX_CURRENT_MODEL to the model you have currently loaded in the server
# To test other models, restart the server with a different model and re-run this script

# Available MLX models to test:
MLX_AVAILABLE_MODELS = [
    "mlx-community/Meta-Llama-3-8B-Instruct-4bit",
    "mlx-community/Qwen2.5-7B-Instruct-4bit",
    "mlx-community/Mistral-7B-Instruct-v0.3-4bit",
]

# Which MLX model is currently loaded in your server?
# Set to None to skip MLX testing, or set to the model name currently running
# Example: mlx_lm.server --model mlx-community/Meta-Llama-3-8B-Instruct-4bit
# MLX_CURRENT_MODEL = None  # Change to model name if MLX server is running
MLX_CURRENT_MODEL = "mlx-community/Meta-Llama-3-8B-Instruct-4bit"  # Uncomment and set your model

# HuggingFace models (via inference API)
HF_MODELS = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.2",
]


# ============================================================
# BENCHMARK FUNCTIONS
# ============================================================

def get_memory_usage() -> float:
    """Get current process memory usage in MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


def warmup_model(agent: Agent, model_name: str) -> bool:
    """Warmup model to avoid cold start bias."""
    print(f"  🔥 Warming up {model_name.split('/')[-1][:30]}...", end=" ", flush=True)
    try:
        # Run a simple warmup query
        agent.run("Hello", stream=False)
        print("✓")
        return True
    except Exception as e:
        print(f"✗ ({str(e)[:30]})")
        return False


def benchmark_model(agent: Agent, model_name: str, provider: str, test_case: Dict) -> BenchmarkResult:
    """Run a single benchmark test"""
    print(f"  Testing: {test_case['name']}...", end=" ", flush=True)

    mem_before = get_memory_usage()
    start_time = time.time()

    try:
        response = agent.run(test_case["prompt"])
        end_time = time.time()
        mem_after = get_memory_usage()

        response_time = end_time - start_time
        response_length = len(response.content) if hasattr(response, 'content') else 0

        # Estimate tokens (rough: ~4 chars per token)
        estimated_tokens = response_length / 4
        tokens_per_second = estimated_tokens / response_time if response_time > 0 else 0

        print(f"✓ ({response_time:.2f}s, {tokens_per_second:.1f} tok/s)")

        return BenchmarkResult(
            model_name=model_name,
            provider=provider,
            task_name=test_case["name"],
            response_time=response_time,
            tokens_per_second=tokens_per_second,
            response_length=response_length,
            success=True,
            memory_used_mb=mem_after - mem_before
        )

    except Exception as e:
        print(f"✗ Error: {str(e)[:50]}")
        return BenchmarkResult(
            model_name=model_name,
            provider=provider,
            task_name=test_case["name"],
            response_time=0,
            tokens_per_second=0,
            response_length=0,
            success=False,
            error=str(e)
        )


def run_ollama_benchmarks(results: List[BenchmarkResult]):
    """Benchmark all Ollama models"""
    print("\n" + "="*60)
    print("🦙 OLLAMA MODELS (Local)")
    print("="*60)

    for model_id in OLLAMA_MODELS:
        print(f"\n📊 Model: {model_id}")
        try:
            agent = Agent(
                model=Ollama(id=model_id),
                markdown=False
            )

            # Warmup to avoid cold start bias
            if not warmup_model(agent, model_id):
                print(f"  ⚠️  Warmup failed, results may be slower than actual")

            for test_case in TEST_CASES:
                result = benchmark_model(agent, model_id, "Ollama", test_case)
                results.append(result)

        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed to initialize: {error_msg[:80]}")
            if "not found" in error_msg.lower() or "pull" in error_msg.lower():
                print(f"  💡 Try: ollama pull {model_id}")


def run_mlx_benchmarks(results: List[BenchmarkResult]):
    """Benchmark MLX model (only one at a time due to server limitation)"""
    print("\n" + "="*60)
    print("🍎 MLX MODELS (Apple Silicon Optimized)")
    print("="*60)

    if MLX_CURRENT_MODEL is None:
        print("⏭️  MLX_CURRENT_MODEL is None - Skipping MLX tests")
        print("   To test MLX models:")
        print("   1. Start server: mlx_lm.server --model <model-name>")
        print("   2. Edit benchmark.py and set MLX_CURRENT_MODEL")
        print(f"   Available models: {', '.join([m.split('/')[-1] for m in MLX_AVAILABLE_MODELS])}")
        return

    print(f"⚠️  Testing single model: {MLX_CURRENT_MODEL}")
    print(f"   (MLX server can only run one model at a time)")
    print(f"\n   To test other models:")
    for model in MLX_AVAILABLE_MODELS:
        if model != MLX_CURRENT_MODEL:
            print(f"   - Restart server: mlx_lm.server --model {model}")
            print(f"     Set MLX_CURRENT_MODEL = \"{model}\"")
            print(f"     Re-run this script")

    print(f"\n📊 Model: {MLX_CURRENT_MODEL}")
    try:
        agent = Agent(
            model=OpenAILike(
                id=MLX_CURRENT_MODEL,
                api_key="not-needed",
                base_url="http://localhost:8080/v1"
            ),
            markdown=False
        )

        # Warmup to avoid cold start bias
        if not warmup_model(agent, MLX_CURRENT_MODEL):
            print(f"  ⚠️  Warmup failed, results may be slower than actual")

        for test_case in TEST_CASES:
            result = benchmark_model(agent, MLX_CURRENT_MODEL, "MLX", test_case)
            results.append(result)

    except Exception as e:
        error_msg = str(e)
        print(f"  ✗ Failed: {error_msg[:80]}")
        if "connection" in error_msg.lower() or "refused" in error_msg.lower():
            print(f"  💡 MLX server not running. Start with:")
            print(f"     mlx_lm.server --model {MLX_CURRENT_MODEL}")
        else:
            print(f"  💡 Make sure the server is running the correct model")


def run_huggingface_benchmarks(results: List[BenchmarkResult]):
    """Benchmark all HuggingFace models"""
    print("\n" + "="*60)
    print("🤗 HUGGINGFACE MODELS (Inference API)")
    print("="*60)
    print("⚠️  Requires HF_TOKEN in .env file")
    print("⚠️  Requires internet connection")
    print("⚠️  May require accepting model licenses on HuggingFace")

    for model_id in HF_MODELS:
        print(f"\n📊 Model: {model_id}")
        try:
            agent = Agent(
                model=HuggingFace(id=model_id, max_tokens=512),
                markdown=False
            )

            # Warmup to avoid cold start bias
            if not warmup_model(agent, model_id):
                print(f"  ⚠️  Warmup failed, results may be slower than actual")

            for test_case in TEST_CASES:
                result = benchmark_model(agent, model_id, "HuggingFace", test_case)
                results.append(result)

        except Exception as e:
            error_msg = str(e)
            print(f"  ✗ Failed: {error_msg[:80]}")
            if "token" in error_msg.lower() or "api_key" in error_msg.lower():
                print(f"  💡 Set HF_TOKEN in .env file")
            elif "gated" in error_msg.lower() or "license" in error_msg.lower():
                print(f"  💡 Accept license at: https://huggingface.co/{model_id}")


# ============================================================
# RESULTS ANALYSIS
# ============================================================

def print_results(results: List[BenchmarkResult]):
    """Print comprehensive results table"""
    print("\n" + "="*80)
    print("📈 BENCHMARK RESULTS - MacBook Pro M1 16GB")
    print("="*80)

    # Filter successful results
    successful = [r for r in results if r.success]

    if not successful:
        print("❌ No successful benchmarks to display")
        return

    # Overall results table
    table_data = []
    for result in successful:
        table_data.append([
            result.provider,
            result.model_name.split('/')[-1][:30],  # Shorten model name
            result.task_name,
            f"{result.response_time:.2f}s",
            f"{result.tokens_per_second:.1f}",
            result.response_length,
        ])

    print("\n📊 Detailed Results:")
    print(tabulate(
        table_data,
        headers=["Provider", "Model", "Task", "Time", "Tok/s", "Length"],
        tablefmt="grid"
    ))

    # Summary by provider
    print("\n📈 Average Performance by Provider:")
    provider_stats = {}
    for provider in ["Ollama", "MLX", "HuggingFace"]:
        provider_results = [r for r in successful if r.provider == provider]
        if provider_results:
            avg_time = sum(r.response_time for r in provider_results) / len(provider_results)
            avg_tokens = sum(r.tokens_per_second for r in provider_results) / len(provider_results)
            provider_stats[provider] = {
                "avg_time": avg_time,
                "avg_tokens": avg_tokens,
                "count": len(provider_results)
            }

    provider_table = [
        [provider, f"{stats['avg_time']:.2f}s", f"{stats['avg_tokens']:.1f}", stats['count']]
        for provider, stats in provider_stats.items()
    ]
    print(tabulate(
        provider_table,
        headers=["Provider", "Avg Time", "Avg Tok/s", "Tests"],
        tablefmt="grid"
    ))

    # Top performers
    print("\n🏆 Top Performers:")
    fastest = max(successful, key=lambda r: r.tokens_per_second)
    print(f"  Fastest: {fastest.model_name} ({fastest.provider}) - {fastest.tokens_per_second:.1f} tok/s")

    # Failed tests
    failed = [r for r in results if not r.success]
    if failed:
        print(f"\n❌ Failed Tests: {len(failed)}")
        for f in failed[:5]:  # Show first 5 failures
            print(f"  - {f.provider}/{f.model_name}: {f.error[:60]}")


# ============================================================
# MAIN
# ============================================================

def main():
    """Run all benchmarks"""
    print("🚀 Starting LLM Benchmark for MacBook Pro M1 16GB")
    print(f"📝 Testing {len(TEST_CASES)} tasks across multiple providers")
    print(f"🎯 Focus: 7-8B models, quality-first, mixed tasks")
    print("\n" + "="*60)
    print("📋 PREREQUISITES:")
    print("  • Ollama: Install with 'brew install ollama' + pull models")
    print("  • MLX: Start server + set MLX_CURRENT_MODEL in this script")
    print(f"    Current: {MLX_CURRENT_MODEL or 'None (skipped)'}")
    print("  • HuggingFace: Set HF_TOKEN in .env file")
    print("="*60)

    input("\n⏸️  Press Enter to start benchmarks (Ctrl+C to cancel)...")

    results: List[BenchmarkResult] = []

    # Run benchmarks for each provider (continue even if one fails)
    try:
        run_ollama_benchmarks(results)
    except Exception as e:
        print(f"\n⚠️  Ollama benchmarks failed: {str(e)}")

    try:
        run_mlx_benchmarks(results)
    except Exception as e:
        print(f"\n⚠️  MLX benchmarks failed: {str(e)}")

    try:
        run_huggingface_benchmarks(results)
    except Exception as e:
        print(f"\n⚠️  HuggingFace benchmarks failed: {str(e)}")

    # Print results
    print_results(results)

    print("\n✅ Benchmark complete!")


if __name__ == "__main__":
    main()

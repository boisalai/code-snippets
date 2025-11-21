"""
Transcribes an audio file in French using the MLX Whisper model and saves the
result to a text file.
See https://github.com/ml-explore/mlx-examples/tree/main/whisper
See also https://simonwillison.net/2024/Aug/13/mlx-whisper/

Installation:
$ brew install ffmpeg
$ uv add mlx-whisper
"""
import time
from pathlib import Path
import mlx_whisper

# Available MLX Whisper models
AVAILABLE_MODELS = {
    "tiny": {
        "repo": "mlx-community/whisper-tiny-mlx",
        "speed": "Fastest",
        "quality": "Basic"
    },
    "base": {
        "repo": "mlx-community/whisper-base-mlx",
        "speed": "Very Fast",
        "quality": "Good"
    },
    "small": {
        "repo": "mlx-community/whisper-small-mlx",
        "speed": "Fast",
        "quality": "Better"
    },
    "medium": {
        "repo": "mlx-community/whisper-medium-mlx",
        "speed": "Moderate",
        "quality": "Very Good"
    },
    "large-v3": {
        "repo": "mlx-community/whisper-large-v3-mlx",
        "speed": "Slower",
        "quality": "Excellent"
    },
    "large-v3-turbo": {
        "repo": "mlx-community/whisper-large-v3-turbo",
        "speed": "Fast",
        "quality": "Excellent (50% faster than large-v3)"
    },
    "distil-large-v3": {
        "repo": "mlx-community/distil-whisper-large-v3",
        "speed": "Fast",
        "quality": "Excellent (30% faster than large-v3)"
    }
}

# Select model here - change this to switch models easily
SELECTED_MODEL = "large-v3-turbo"  # Options: tiny, base, small, medium, large-v3, large-v3-turbo, distil-large-v3

start_time = time.time()

model = AVAILABLE_MODELS[SELECTED_MODEL]["repo"]
print(f"Using model: {SELECTED_MODEL}")
print(f"  Repository: {model}")
print(f"  Speed: {AVAILABLE_MODELS[SELECTED_MODEL]['speed']}")
print(f"  Quality: {AVAILABLE_MODELS[SELECTED_MODEL]['quality']}\n")

downloads_folder = Path.home() / "Downloads"
audio_file_path = str(downloads_folder / "Audio_11_21_2025_13_23_42.mp3")
txt_file_path = str(downloads_folder / "Transcription.txt")

print(audio_file_path)

try:
    result = mlx_whisper.transcribe(
        audio_file_path,
        path_or_hf_repo=model,
        language="fr",
        word_timestamps=True
    )

    # Save transcription to file
    with open(txt_file_path, 'w', encoding='utf-8') as f:
        f.write(result['text'])

    print(f"\nTranscription saved to: {txt_file_path}")

except FileNotFoundError:
    print(f"Le fichier {audio_file_path} n'a pas été trouvé.")
except Exception as e:
    print(f"Une erreur s'est produite lors de la transcription : {e}")

end_time = time.time()
print(f"Execution time: {(end_time - start_time)/60:.1f} minutes")
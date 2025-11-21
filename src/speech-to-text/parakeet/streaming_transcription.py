"""
See https://github.com/senstella/parakeet-mlx
uv add parakeet-mlx -U
"""
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio
import numpy as np

model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

# Create a streaming context
with model.transcribe_stream(
    context_size=(256, 256),  # (left_context, right_context) frames
) as transcriber:
    # Simulate real-time audio chunks
    audio_data = load_audio("audio_file.wav", model.preprocessor_config.sample_rate)
    chunk_size = model.preprocessor_config.sample_rate  # 1 second chunks

    for i in range(0, len(audio_data), chunk_size):
        chunk = audio_data[i:i+chunk_size]
        transcriber.add_audio(chunk)

        # Access current transcription
        result = transcriber.result
        print(f"Current text: {result.text}")

        # Access finalized and draft tokens
        # transcriber.finalized_tokens
        # transcriber.draft_tokens
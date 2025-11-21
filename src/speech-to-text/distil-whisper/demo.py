"""
Pas testé 
https://huggingface.co/distil-whisper/distil-large-v3/tree/main
"""

from transformers import pipeline

transcriber = pipeline("automatic-speech-recognition", 
                      model="distil-whisper/distil-large-v3")

# Détection automatique de la langue
audio_file = "/Users/alain/Downloads/Recording_2.mp3"
result = transcriber(audio_file, generate_kwargs={"language": "french"})
# ou language="english"

print(result)
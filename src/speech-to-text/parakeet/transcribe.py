"""
See https://github.com/senstella/parakeet-mlx
uv add parakeet-mlx -U

Ce code retranscrit un fichier audio mais en anglais, donc pas très pratique.

Usage:
# Basic usage (saves in the same folder as the audio)
transcribe_audio("/path/to/audio.mp3")

# With custom chunk_duration to reduce memory usage
transcribe_audio("/path/to/audio.mp3", chunk_duration=60.0)

# With custom output directory
transcribe_audio("/path/to/audio.mp3", output_dir="/path/to/output")
"""
from pathlib import Path
from parakeet_mlx import from_pretrained


def transcribe_audio(audio_file_path: str, chunk_duration: float = 120.0, output_dir: str = None):
    """
    Transcribe an audio file using Parakeet MLX model.

    Args:
        audio_file_path: Path to the audio file to transcribe
        chunk_duration: Duration in seconds for processing chunks (default: 120.0)
        output_dir: Directory to save the transcription (default: same as audio file)

    Returns:
        str: Transcribed text or None if error occurred
    """
    try:
        # Load model
        print(f"Loading model...")
        model = from_pretrained("mlx-community/parakeet-tdt-0.6b-v3")

        # Verify audio file exists
        audio_path = Path(audio_file_path)
        if not audio_path.exists():
            print(f"Error: Audio file not found: {audio_file_path}")
            return None

        print(f"Transcribing: {audio_path.name}")

        # Transcribe with chunk processing to avoid memory issues
        result = model.transcribe(
            str(audio_path),
            chunk_duration=chunk_duration
        )

        # Determine output file path
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            txt_file_path = output_path / f"{audio_path.stem}.txt"
        else:
            txt_file_path = audio_path.with_suffix('.txt')

        # Save transcription to file
        with open(txt_file_path, 'w', encoding='utf-8') as f:
            f.write(result.text)

        print(f"Transcription saved to: {txt_file_path}")
        return result.text

    except RuntimeError as e:
        if "metal::malloc" in str(e):
            print(f"Memory error: Try reducing chunk_duration (current: {chunk_duration}s)")
            print(f"Suggested: {chunk_duration / 2}s")
        else:
            print(f"Runtime error: {e}")
        return None
    except Exception as e:
        print(f"Error during transcription: {e}")
        return None


if __name__ == "__main__":
    # Example usage
    audio_file = "/Users/alain/Downloads/Audio_11_21_2025_13_23_42.mp3"
    transcription = transcribe_audio(audio_file, chunk_duration=120.0)

    if transcription:
        print("\n--- Transcription ---")
        print(transcription)

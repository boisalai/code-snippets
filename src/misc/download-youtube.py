import yt_dlp
from pathlib import Path
from typing import Union, List, Literal

def download_youtube_video(
    url: Union[str, List[str]], 
    output_path: Union[str, Path] = None,
    filename: str = "video.mp4"
) -> Path:
    """
    Downloads a YouTube video to the specified location.
    
    Args:
        url: YouTube video URL or list of URLs
        output_path: Directory where to save the video (default: ~/Downloads)
        filename: Name of the output file (default: video.mp4)
        
    Returns:
        Path: Path to the downloaded video file
        
    Raises:
        ValueError: If the output path is not a directory
        Exception: If download fails
    """
    # Convert single URL to list
    if isinstance(url, str):
        url = [url]
    
    # Set default output path
    if output_path is None:
        output_path = Path.home() / "Downloads"
    else:
        output_path = Path(output_path).expanduser().resolve()
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not output_path.is_dir():
        raise ValueError(f"{output_path} is not a directory")
    
    # Create full output file path
    output_file = output_path / filename
    
    # Download options
    opts = {
        'outtmpl': str(output_file),
        'format': 'best',
    }
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.download(url)
        print(f"Video downloaded: {output_file}")
        return output_file
    except Exception as e:
        raise Exception(f"Download failed: {e}")


def download_youtube_audio(
    url: Union[str, List[str]], 
    output_path: Union[str, Path] = None,
    filename: str = "%(title)s.%(ext)s",
    audio_format: Literal['mp3', 'wav', 'aac', 'm4a', 'flac'] = 'mp3',
    audio_quality: str = '192'
) -> Path:
    """
    Downloads audio from a YouTube video.
    
    Args:
        url: YouTube video URL or list of URLs
        output_path: Directory where to save the audio (default: ~/Downloads)
        filename: Output filename pattern (default: uses video title)
                 Use %(title)s for video title, %(ext)s for extension
        audio_format: Output audio format (mp3, wav, aac, m4a, flac)
        audio_quality: Audio quality in kbps (128, 192, 256, 320)
        
    Returns:
        Path: Path to the downloaded audio file
        
    Raises:
        ValueError: If the output path is not a directory
        Exception: If download fails or ffmpeg is not installed
    """
    # Convert single URL to list
    if isinstance(url, str):
        url = [url]
    
    # Set default output path
    if output_path is None:
        output_path = Path.home() / "Downloads"
    else:
        output_path = Path(output_path).expanduser().resolve()
    
    # Ensure output directory exists
    output_path.mkdir(parents=True, exist_ok=True)
    
    if not output_path.is_dir():
        raise ValueError(f"{output_path} is not a directory")
    
    # Create full output template path
    output_template = output_path / filename
    
    # Download options
    opts = {
        'outtmpl': str(output_template),
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': audio_format,
            'preferredquality': audio_quality,
        }],
    }
    
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url[0], download=True)
            # Get the actual filename after download
            if 'title' in info:
                downloaded_file = output_path / f"{info['title']}.{audio_format}"
                print(f"Audio downloaded: {downloaded_file}")
                return downloaded_file
            else:
                print(f"Audio downloaded to: {output_path}")
                return output_path
    except Exception as e:
        raise Exception(f"Download failed: {e}")


if __name__ == "__main__":
    # Download video examples
    # Simple usage with default location
    # video_url = "https://www.youtube.com/watch?v=0IhZdcjddo4"
    # output = download_youtube_video(video_url)

    # Custom location and filename
    # output = download_youtube_video(
    #     url=video_url,
    #     output_path=Path("~/Videos/YouTube"),
    #     filename="my_video.mp4"
    # )

    # Multiple videos
    # urls = [
    #     "https://www.youtube.com/watch?v=0IhZdcjddo4",
    #     "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    # ]
    # output = download_youtube_video(urls, filename="playlist_%(title)s.mp4")

    # Download audio examples
    # Simple usage with default settings (MP3, 192kbps)
    audio_url = "https://www.youtube.com/watch?v=eC3RNuI6ow0"
    output = download_youtube_audio(audio_url)

    # Custom location and format
    # output = download_youtube_audio(
    #     url=audio_url,
    #     output_path=Path("~/Music/YouTube"),
    #     audio_format='m4a',
    #     audio_quality='256'
    # )

    # High quality FLAC
    # output = download_youtube_audio(
    #     url=audio_url,
    #     audio_format='flac',
    #     audio_quality='320'
    # )

    # Custom filename
    # output = download_youtube_audio(
    #     url=audio_url,
    #     filename='my_audio.%(ext)s'
    # )

    # Multiple videos
    # urls = [
    #     "https://www.youtube.com/watch?v=eC3RNuI6ow0",
    #     "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    # ]
    # for url in urls:
    #     download_youtube_audio(url)
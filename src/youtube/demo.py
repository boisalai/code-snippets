import yt_dlp

url = "https://www.youtube.com/watch?v=MSizsAm2eS4"

options = {
    'format': 'bestaudio/best',
    'cookiesfrombrowser': ('firefox',),  # ou 'chrome', 'safari', 'edge'
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192',
    }],
    'outtmpl': '%(title)s.%(ext)s',
    'extractor_args': {
        'youtube': {
            'player_client': ['default', '-android_sdkless'],
        }
    },
}

with yt_dlp.YoutubeDL(options) as ydl:
    ydl.download([url])
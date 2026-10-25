
import requests
import os


def text_to_speech(text, output_path, language="te-IN"):
    api_key = os.environ.get("GNANI_API_KEY")

    if not api_key:
        raise RuntimeError("GNANI_API_KEY is not configured.")

    url = "https://api.vachana.ai/api/v1/tts/inference"

    headers = {
        "X-API-Key-ID": api_key,
        "Content-Type": "application/json"
    }

    payload = {
        "text": text,
        "model": "timbre-v2.5",
        "voice": "Pranav",
        "language": language,
        "audio_config": {
            "container": "mp3",
            "sample_rate": 44100,
            "bitrate": "128k"
        }
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    if "audio" not in response.headers.get("content-type", "").lower():
        raise ValueError("The TTS API did not return audio.")

    with open(output_path, "wb") as audio_file:
        audio_file.write(response.content)

    return output_path

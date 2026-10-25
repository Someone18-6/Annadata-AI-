
import requests
import os


def speech_to_text(audio_path, language="te-IN"):
    api_key = os.environ.get("GNANI_API_KEY")

    if not api_key:
        raise RuntimeError("GNANI_API_KEY is not configured.")

    url = "https://api.vachana.ai/stt/v3"

    headers = {
        "X-API-Key-ID": api_key
    }

    data = {
        "language_code": language,
        "preferred_language": language,
        "format": "transcribe",
        "itn_native_numerals": "true"
    }

    with open(audio_path, "rb") as audio_file:
        files = {
            "audio_file": (
                os.path.basename(audio_path),
                audio_file,
                "audio/wav"
            )
        }

        response = requests.post(
            url,
            headers=headers,
            data=data,
            files=files,
            timeout=120
        )

    response.raise_for_status()
    result = response.json()

    if not result.get("success"):
        raise RuntimeError(f"Transcription failed: {result}")

    return result["transcript"]

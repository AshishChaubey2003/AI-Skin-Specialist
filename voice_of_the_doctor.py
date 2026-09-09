import os
import platform
import subprocess
from pathlib import Path

from deepgram import DeepgramClient
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DOCTOR_AUDIO = BASE_DIR / "doctor_response.mp3"


def convert_text_to_doctor_audio(text, output_filepath=DEFAULT_DOCTOR_AUDIO):
    deepgram_api_key = os.environ.get("DEEPGRAM_API_KEY")
    if not deepgram_api_key:
        raise ValueError("Missing DEEPGRAM_API_KEY in .env or environment")

    deepgram = DeepgramClient(api_key=deepgram_api_key)

    # Deepgram TTS rejects text over 2000 characters - trim safely if needed.
    MAX_TTS_CHARS = 1900
    if len(text) > MAX_TTS_CHARS:
        text = text[:MAX_TTS_CHARS].rsplit(".", 1)[0] + "."

    audio = deepgram.speak.v1.audio.generate(
        text=text,
        model=os.environ.get("DEEPGRAM_TTS_MODEL", "aura-2-thalia-en"),
        encoding="mp3",
    )

    output_filepath = Path(output_filepath)
    with output_filepath.open("wb") as file:
        for chunk in audio:
            file.write(chunk)

    return output_filepath


def play_audio(audio_filepath):
    audio_filepath = str(audio_filepath)

    if platform.system() == "Darwin":
        subprocess.run(["afplay", audio_filepath], check=False)
    elif platform.system() == "Windows":
        os.startfile(audio_filepath)
    else:
        subprocess.run(["xdg-open", audio_filepath], check=False)


if __name__ == "__main__":
    print("Starting TTS test...")
    text = "Hi, my name is AI with Hassan, who are you? I am very happy."
    audio_path = convert_text_to_doctor_audio(text)
    print(f"Audio saved to: {audio_path}")
    play_audio(audio_path)
import base64
import os
from io import BytesIO

import cv2
from dotenv import load_dotenv
from groq import Groq
from PIL import Image


load_dotenv()


def encode_image_for_groq(filepath_or_pil_image):
    """Resize + convert image (file path or PIL Image) to base64 JPEG for Groq vision."""
    if isinstance(filepath_or_pil_image, Image.Image):
        image = filepath_or_pil_image
    else:
        image = Image.open(filepath_or_pil_image)

    image.thumbnail((1024, 1024))

    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=75)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_frame_from_video(video_filepath):
    """
    Grabs a single representative frame from the middle of the video
    and returns it as a PIL Image. Returns None if extraction fails.
    """
    try:
        cap = cv2.VideoCapture(video_filepath)
        if not cap.isOpened():
            return None

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        middle_frame_index = max(total_frames // 2, 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, middle_frame_index)

        success, frame = cap.read()
        cap.release()

        if not success:
            return None

        # OpenCV gives BGR, PIL expects RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)
    except Exception:
        return None


def brain_of_the_doctor(patient_text, image_filepath=None, video_filepath=None):
    """
    Sends patient's text + skin image (and, if available, a frame extracted
    from the video) to Groq vision model and returns a short, doctor-style
    response (used later for text-to-speech).
    """
    groq_api_key = os.environ.get("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("Missing GROQ_API_KEY in .env or environment")

    if not image_filepath and not video_filepath:
        raise ValueError("Please provide an image and/or a video of the skin concern.")

    content = []
    video_frame_included = False
    frame = None

    prompt = (
        "You are a confident, natural doctor specializing in skin care. Speak with the reassurance, clarity, and authority of a real doctor. "
        "Limit your entire response to two or three sentences maximum. "
        "Do not use any special characters, symbols, asterisks, or markdown formatting in your response because it will be converted directly to audio.\n\n"
        f"Patient text: {patient_text}"
    )

    if image_filepath:
        prompt += "\nThe first image is the patient's uploaded photo."

    if video_filepath:
        frame = extract_frame_from_video(video_filepath)
        if frame is not None:
            video_frame_included = True
            prompt += (
                "\nAn additional image is a single frame extracted from the patient's uploaded video. "
                "This is not the full video, just one moment from it, so mention that you are basing "
                "your assessment on a snapshot from the video rather than the full clip."
            )
        else:
            prompt += "\nThe patient also uploaded a video, but a frame could not be extracted from it."

    content.append({"type": "text", "text": prompt})

    if image_filepath:
        image_data = encode_image_for_groq(image_filepath)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_data}"},
            }
        )

    if video_frame_included:
        frame_data = encode_image_for_groq(frame)
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{frame_data}"},
            }
        )

    client = Groq(api_key=groq_api_key)
    response = client.chat.completions.create(
        model=os.environ.get("GROQ_MODEL", "qwen/qwen3.6-27b"),
        max_completion_tokens=1000,
        reasoning_effort="none",
        messages=[
            {
                "role": "system",
                "content": "You are a careful skin care assistant. Give general information, not a diagnosis.",
            },
            {
                "role": "user",
                "content": content,
            },
        ],
    )

    raw_response = response.choices[0].message.content

    # Qwen model sometimes wraps its reasoning in <think>...</think> tags.
    # Strip any such block robustly (handles missing opening/closing tags too).
    import re

    cleaned = re.sub(r"<think>.*?</think>", "", raw_response, flags=re.DOTALL)
    if "</think>" in cleaned:
        cleaned = cleaned.split("</think>")[-1]
    if "<think>" in cleaned:
        # Opening tag with no closing tag - drop everything from it onward
        cleaned = cleaned.split("<think>")[0]

    cleaned = cleaned.strip()

    # Safety net: Deepgram TTS rejects text over 2000 characters.
    MAX_TTS_CHARS = 1900
    if len(cleaned) > MAX_TTS_CHARS:
        cleaned = cleaned[:MAX_TTS_CHARS].rsplit(".", 1)[0] + "."

    # Safety net: if stripping <think> left nothing (model ran out of tokens
    # mid-thought), fall back to a generic message instead of sending an
    # empty string to the TTS engine.
    if not cleaned:
        cleaned = (
            "I was unable to complete my assessment from the provided input. "
            "Please try again or consult a licensed dermatologist directly."
        )

    return cleaned


if __name__ == "__main__":
    print("Starting test...")
    result = brain_of_the_doctor(
        "I have some redness on my arm",
        image_filepath="sample-image.png",
        video_filepath="test-video.mp4",
    )
    print("Got result:")
    print(result)
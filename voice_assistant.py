# Mic → listen_once() → text
# text + screenshot → Gemini → answer
# answer → speak() → TTS output
#          ↑ barge-in kills this if you

import speech_recognition as sr
import pyautogui
import time
from PIL import Image
from google import genai
import pyttsx3
import re
import numpy as np
import threading
import pyaudio
import noisereduce as nr
import subprocess
import sys

GEMINI_API_KEY = "AIzaSyB5W_K4gBBhHLe-HxUMO92BFOd2ui8HU7U"

client = genai.Client(api_key=GEMINI_API_KEY)


def get_working_model():
    preferred = [
        "models/gemini-2.5-flash",
        "models/gemini-2.0-flash",
        "models/gemini-flash-latest"
    ]
    available = [m.name for m in client.models.list()]
    print("Available models:", available)
    for p in preferred:
        if p in available:
            return p
    for name in available:
        if "flash" in name and "image" not in name:
            return name
    return available[0]


MODEL_NAME = get_working_model()
print("Using Gemini model:", MODEL_NAME)


BARGE_IN_THRESHOLD = 2000
BARGE_IN_CHUNK     = 512
BARGE_IN_RATE      = 16000


def clean_for_speech(text):
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    text = re.sub(r"`[^`]*`", "", text)
    text = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", text)
    text = re.sub(r"_{1,3}(.*?)_{1,3}", r"\1", text)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*•]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    text = text.replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text



_TTS_WORKER_SCRIPT = """
import pyttsx3, sys
text = sys.argv[1]
eng = pyttsx3.init()
eng.setProperty('rate', 170)
eng.say(text)
eng.runAndWait()
eng.stop()
"""


def speak(text, enable_barge_in=True):
    """
    Speak text in a child process.
    While speaking, monitor mic volume in a thread.
    If user starts talking → kill the child process instantly → return True.
    Returns False if speech finished normally.
    """
    if not text or not text.strip():
        return False

    text = clean_for_speech(text)
    if not text:
        return False

    print(f"[speak] Speaking: {text[:80]}...")

    import tempfile, os
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False)
    tmp.write(_TTS_WORKER_SCRIPT)
    tmp.close()

    tts_proc = subprocess.Popen(
        [sys.executable, tmp.name, text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

    interrupted = False

    if enable_barge_in:

        p = pyaudio.PyAudio()
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=BARGE_IN_RATE,
            input=True,
            frames_per_buffer=BARGE_IN_CHUNK
        )

        try:
            while tts_proc.poll() is None:   # while TTS is still running
                data = stream.read(BARGE_IN_CHUNK, exception_on_overflow=False)
                volume = np.abs(np.frombuffer(data, dtype=np.int16)).mean()

                if volume > BARGE_IN_THRESHOLD:
                    print(f"\n[Barge-in] Detected (volume={volume:.0f}) — killing TTS.")
                    tts_proc.kill()
                    tts_proc.wait()
                    interrupted = True
                    break

                time.sleep(0.01)

        except Exception as e:
            print(f"[Barge-in monitor error] {e}")

        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
    else:
        tts_proc.wait()

    try:
        os.unlink(tmp.name)
    except Exception:
        pass

    return interrupted


def reduce_noise_from_audio(audio: sr.AudioData) -> sr.AudioData:
    sample_rate = audio.sample_rate
    raw_data = np.frombuffer(audio.get_raw_data(), dtype=np.int16).astype(np.float32)
    noise_sample_length = int(sample_rate * 0.5)
    noise_sample = raw_data[:noise_sample_length]

    reduced = nr.reduce_noise(
        y=raw_data,
        y_noise=noise_sample,
        sr=sample_rate,
        prop_decrease=0.9,
        stationary=False,
    )

    reduced_bytes = reduced.astype(np.int16).tobytes()
    return sr.AudioData(reduced_bytes, sample_rate, audio.sample_width)


recognizer = sr.Recognizer()
mic = sr.Microphone()

recognizer.energy_threshold         = 3000   
recognizer.dynamic_energy_threshold = False 
recognizer.pause_threshold          = 0.8
recognizer.phrase_threshold         = 0.3
recognizer.non_speaking_duration    = 0.5

MIN_ENERGY_THRESHOLD = 2000


def listen_once():
    try:
        with mic as source:
            print("Calibrating for background noise...")
            recognizer.adjust_for_ambient_noise(source, duration=1.5)

            if recognizer.energy_threshold < MIN_ENERGY_THRESHOLD:
                recognizer.energy_threshold = MIN_ENERGY_THRESHOLD

            print(f"Energy threshold: {recognizer.energy_threshold:.0f}")
            print("Listening...")

            # WaitTimeoutError can happen here — must be inside try
            audio = recognizer.listen(
                source,
                timeout=10,
                phrase_time_limit=10
            )

    except sr.WaitTimeoutError:
        print("No speech detected — timed out waiting. Will retry.")
        return ""
    except Exception as e:
        print(f"Microphone error: {e}")
        return ""

    try:
        print("Reducing background noise...")
        clean_audio = reduce_noise_from_audio(audio)
        result = recognizer.recognize_google(clean_audio, language="en-US")
        return result

    except sr.UnknownValueError:
        print("Could not understand audio.")
        return ""
    except sr.RequestError as e:
        print(f"Speech recognition error: {e}")
        return ""
    except Exception as e:
        print(f"listen_once error: {e}")
        return ""

# Screenshot

def take_slide_screenshot():
    filename = f"slide_{int(time.time())}.png"
    img = pyautogui.screenshot()
    img.save(filename)
    print(f"Screenshot saved: {filename}")
    return filename


def explain_slide_with_gemini(image_path, user_text):
    img = Image.open(image_path)

    prompt = f"""
You are helping during a live presentation.

The speaker said:
{user_text}

Look at the slide image and explain what is shown.
Speak naturally like a presenter — use plain sentences only.
Do NOT use bullet points, markdown, bold, or any special formatting.
Keep your answer under 80 words.
Do not say you are an AI.
"""

    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=[prompt, img]
    )

    if response and response.text:
        return response.text.strip()
    return ""

def check_voice_command():
    text = listen_once()

    if not text.strip():
        return

    print("User said:", text)

    speak("Ok.")

    image_path = take_slide_screenshot()

    try:
        answer = explain_slide_with_gemini(image_path, text)
        print("Gemini answer:", answer)
    except Exception as e:
        print("Gemini error:", e)
        speak("Sorry, I could not read this slide.")
        return

    if answer:
        interrupted = speak(answer)
        if interrupted:
            print("[Barge-in] User interrupted — restarting listen.")
            check_voice_command()
    else:
        speak("I could not understand this slide clearly.")
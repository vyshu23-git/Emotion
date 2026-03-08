import sounddevice as sd
import soundfile as sf
from faster_whisper import WhisperModel
from transformers import pipeline

SAMPLERATE = 16000
DURATION = 3
AUDIO_FILE = "voice.wav"

whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
emotion_model = pipeline("text-classification",
                         model="j-hartmann/emotion-english-distilroberta-base")

def detect_speech_emotion(_):

    audio = sd.rec(int(DURATION * SAMPLERATE),
                   samplerate=SAMPLERATE,
                   channels=1)

    sd.wait()
    sf.write(AUDIO_FILE, audio, SAMPLERATE)

    segments, _ = whisper_model.transcribe(AUDIO_FILE)

    text = " ".join(segment.text for segment in segments)

    if text == "":
        return "No speech detected", 0

    result = emotion_model(text)

    emotion = result[0]['label']
    score = result[0]['score']

    return emotion, score
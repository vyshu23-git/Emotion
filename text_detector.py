from transformers import pipeline

emotion_model = pipeline(
    "text-classification",
    model="j-hartmann/emotion-english-distilroberta-base"
)

def detect_text_emotion(text):

    result = emotion_model(text)

    emotion = result[0]['label']
    score = result[0]['score']

    return emotion, score
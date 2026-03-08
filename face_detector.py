from deepface import DeepFace
import cv2
import numpy as np
from PIL import Image

def detect_face_emotion(img):

    # Convert uploaded image to array
    image = Image.open(img)
    frame = np.array(image)

    try:
        result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)

        emotion = result[0]['dominant_emotion']
        score = max(result[0]['emotion'].values())

        return emotion, score

    except:
        return "No face detected", 0
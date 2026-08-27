import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
from mediapipe.python.solutions import face_mesh as mp_face_mesh

# (Elimina la línea anterior que decía: mp_face_mesh = mp.solutions.face_mesh)

class MemeTrackerProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extraer puntos: Comisuras (61, 291) y Labios (13, 14)
                c1 = np.array([face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h])
                c2 = np.array([face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h])
                l1 = np.array([face_landmarks.landmark[13].x * w, face_landmarks.landmark[13].y * h])
                l2 = np.array([face_landmarks.landmark[14].x * w, face_landmarks.landmark[14].y * h])

                # Distancia euclidiana
                ancho_boca = np.linalg.norm(c1 - c2)
                alto_boca = np.linalg.norm(l1 - l2)

                # Clasificación de expresión
                if alto_boca > 25:
                    meme_text = "😲 Gato Sorprendido"
                    color = (0, 255, 255)
                elif ancho_boca > 80:
                    meme_text = "😁 Gato Sonriente"
                    color = (0, 255, 0)
                else:
                    meme_text = "😐 Gato Juzgando"
                    color = (255, 255, 255)

                # Overlay en pantalla
                cv2.rectangle(img, (20, 20), (380, 70), (0, 0, 0), -1)
                cv2.putText(img, meme_text, (30, 55),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# UI de Streamlit
st.title("🐱 Tracker de Memes VTuber")
st.write("Presiona 'Start' para encender la cámara y probar tus expresiones.")

webrtc_streamer(
    key="meme-tracker",
    video_processor_factory=MemeTrackerProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

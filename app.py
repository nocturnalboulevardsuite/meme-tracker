import av
import cv2
import numpy as np
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Función auxiliar para descargar una imagen de meme desde una URL
def cargar_imagen_url(url):
    try:
        # Añadido un User-Agent para evitar que servidores bloqueen la solicitud por ser un script
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None

# Función para pegar una imagen (con o sin transparencia PNG) sobre la pantalla
def superponer_imagen(fondo, overlay, x, y, size=None):
    if overlay is None:
        return fondo
    if size:
        overlay = cv2.resize(overlay, size)

    h_f, w_f, _ = fondo.shape
    h_o, w_o = overlay.shape[:2]

    x1, x2 = max(0, x), min(w_f, x + w_o)
    y1, y2 = max(0, y), min(h_f, y + h_o)

    ox1, ox2 = max(0, -x), min(w_o, w_f - x)
    oy1, oy2 = max(0, -y), min(h_o, h_f - y)

    if x1 >= x2 or y1 >= y2 or ox1 >= ox2 or oy1 >= oy2:
        return fondo

    if overlay.shape[2] == 4:  # Si el PNG tiene fondo transparente
        alpha = overlay[oy1:oy2, ox1:ox2, 3] / 255.0
        for c in range(3):
            fondo[y1:y2, x1:x2, c] = (
                alpha * overlay[oy1:oy2, ox1:ox2, c] +
                (1.0 - alpha) * fondo[y1:y2, x1:x2, c]
            )
    else:
        fondo[y1:y2, x1:x2] = overlay[oy1:oy2, ox1:ox2, :3]

    return fondo

# IMPORTANTE: Reemplaza esta URL por un enlace directo a un archivo de imagen real (.png o .jpg)
URL_MEME_SORPRENDIDO = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/200px-SNice.svg.png"

class MemeTrackerProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        # Corrección: Se invoca la función en lugar de reasignarla a una cadena de texto
        self.img_sorprendido = cargar_imagen_url(URL_MEME_SORPRENDIDO)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w, _ = img.shape
        
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_img)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # 1. Dibujar la malla de detección en verde (líneas verdes)
                mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELLATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=1, circle_radius=1
                    )
                )

                # 2. Obtener puntos clave para gestos
                # Boca
                c1 = np.array([face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h])
                c2 = np.array([face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h])
                l1 = np.array([face_landmarks.landmark[13].x * w, face_landmarks.landmark[13].y * h])
                l2 = np.array([face_landmarks.landmark[14].x * w, face_landmarks.landmark[14].y * h])

                # Ceja y Ojo izquierdo (para detectar fruncido / enojado)
                ceja_izq = np.array([face_landmarks.landmark[70].x * w, face_landmarks.landmark[70].y * h])
                ojo_izq = np.array([face_landmarks.landmark[159].x * w, face_landmarks.landmark[159].y * h])

                ancho_boca = np.linalg.norm(c1 - c2)
                alto_boca = np.linalg.norm(l1 - l2)
                dist_ceja_ojo = np.linalg.norm(ceja_izq - ojo_izq)

                # Centro de la cara para colocar la imagen
                centro_x = int(face_landmarks.landmark[1].x * w)
                centro_y = int(face_landmarks.landmark[1].y * h)

                # 3. Lógica de estados y detección de emociones
                if dist_ceja_ojo < 15:  # Ceja muy cerca del ojo
                    meme_text = "ENOJADO"
                    color = (0, 0, 255)  # Rojo
                elif alto_boca > 25:
                    meme_text = "SORPRENDIDO"
                    color = (0, 255, 255)  # Amarillo
                    
                    # Superponer meme PNG centrado en la cara
                    img = superponer_imagen(
                        img, 
                        self.img_sorprendido, 
                        x=centro_x - 100, 
                        y=centro_y - 100, 
                        size=(200, 200)
                    )
                elif ancho_boca > 80:
                    meme_text = "SONRIENTE"
                    color = (0, 255, 0)  # Verde
                else:
                    meme_text = "JUZGANDO"
                    color = (255, 255, 255)

                # 4. Cuadro de texto superior
                cv2.rectangle(img, (20, 20), (320, 70), (0, 0, 0), -1)
                cv2.putText(img, meme_text, (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🐱 Tracker de Memes VTuber")
st.write("Presiona 'Start' para encender la cámara y probar tus expresiones.")

webrtc_streamer(
    key="meme-tracker",
    video_processor_factory=MemeTrackerProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

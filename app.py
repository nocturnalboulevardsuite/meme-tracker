import av
import cv2
import numpy as np
import urllib.request
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import mediapipe as mp
import traceback

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

# Función auxiliar para descargar una imagen de meme desde una URL
def cargar_imagen_url(url):
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        response = urllib.request.urlopen(req)
        arr = np.asarray(bytearray(response.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
        return img
    except Exception as e:
        print(f"Error cargando imagen: {e}")
        return None

# Función optimizada y segura para pegar una imagen PNG sobre la pantalla
def superponer_imagen(fondo, overlay, x, y, size=None):
    if overlay is None:
        return fondo
    if size:
        overlay = cv2.resize(overlay, size)

    h_f, w_f = fondo.shape[:2]
    h_o, w_o = overlay.shape[:2]

    # Calcular límites de la imagen y asegurar que no se salga de la pantalla
    x1, x2 = max(0, x), min(w_f, x + w_o)
    y1, y2 = max(0, y), min(h_f, y + h_o)

    ox1, ox2 = max(0, -x), min(w_o, w_f - x)
    oy1, oy2 = max(0, -y), min(h_o, h_f - y)

    # Si la imagen está completamente fuera de la pantalla, no hacer nada
    if x1 >= x2 or y1 >= y2 or ox1 >= ox2 or oy1 >= oy2:
        return fondo

    # Extraer las regiones de interés (ROI) de ambas imágenes
    fondo_roi = fondo[y1:y2, x1:x2]
    overlay_roi = overlay[oy1:oy2, ox1:ox2]

    if overlay.shape[2] == 4:  # Si tiene transparencia
        alpha = overlay_roi[:, :, 3] / 255.0
        alpha = np.expand_dims(alpha, axis=-1)
        
        # Combinar usando el canal alfa y forzar tipo uint8
        res = (alpha * overlay_roi[:, :, :3] + (1 - alpha) * fondo_roi).astype(np.uint8)
        fondo[y1:y2, x1:x2] = res
    else:
        fondo[y1:y2, x1:x2] = overlay_roi[:, :, :3]

    return fondo

# URL directa a un archivo de imagen (reemplazar por la tuya si lo deseas)
URL_MEME_SORPRENDIDO = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/SNice.svg/200px-SNice.svg.png"

class MemeTrackerProcessor(VideoProcessorBase):
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.img_sorprendido = cargar_imagen_url(URL_MEME_SORPRENDIDO)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        # 1. Convertir el frame entrante
        img = frame.to_ndarray(format="bgr24")
        
        # 2. Envolver todo en try-except para evitar que la cámara se congele
        try:
            h, w, _ = img.shape
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_img)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Dibujar malla
                    mp_drawing.draw_landmarks(
                        image=img,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_TESSELLATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing.DrawingSpec(
                            color=(0, 255, 0), thickness=1, circle_radius=1
                        )
                    )

                    # Puntos clave
                    c1 = np.array([face_landmarks.landmark[61].x * w, face_landmarks.landmark[61].y * h])
                    c2 = np.array([face_landmarks.landmark[291].x * w, face_landmarks.landmark[291].y * h])
                    l1 = np.array([face_landmarks.landmark[13].x * w, face_landmarks.landmark[13].y * h])
                    l2 = np.array([face_landmarks.landmark[14].x * w, face_landmarks.landmark[14].y * h])

                    ceja_izq = np.array([face_landmarks.landmark[70].x * w, face_landmarks.landmark[70].y * h])
                    ojo_izq = np.array([face_landmarks.landmark[159].x * w, face_landmarks.landmark[159].y * h])

                    ancho_boca = np.linalg.norm(c1 - c2)
                    alto_boca = np.linalg.norm(l1 - l2)
                    dist_ceja_ojo = np.linalg.norm(ceja_izq - ojo_izq)

                    centro_x = int(face_landmarks.landmark[1].x * w)
                    centro_y = int(face_landmarks.landmark[1].y * h)

                    # Lógica de emociones
                    if dist_ceja_ojo < 15:
                        meme_text = "ENOJADO"
                        color = (0, 0, 255)
                    elif alto_boca > 25:
                        meme_text = "SORPRENDIDO"
                        color = (0, 255, 255)
                        img = superponer_imagen(
                            img, 
                            self.img_sorprendido, 
                            x=centro_x - 100, 
                            y=centro_y - 100, 
                            size=(200, 200)
                        )
                    elif ancho_boca > 80:
                        meme_text = "SONRIENTE"
                        color = (0, 255, 0)
                    else:
                        meme_text = "JUZGANDO"
                        color = (255, 255, 255)

                    cv2.rectangle(img, (20, 20), (320, 70), (0, 0, 0), -1)
                    cv2.putText(img, meme_text, (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    
        except Exception as e:
            # Si ocurre CUALQUIER error en este fotograma, lo imprime en consola y sigue con el video normal
            print(f"Error procesando frame: {e}")
            traceback.print_exc()

        # 3. Devolver imagen procesada (o la original intacta si falló algo arriba)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

st.title("🐱 Tracker de Memes VTuber")
st.write("Presiona 'Start' para encender la cámara y probar tus expresiones.")

webrtc_streamer(
    key="meme-tracker",
    video_processor_factory=MemeTrackerProcessor,
    media_stream_constraints={"video": True, "audio": False},
)

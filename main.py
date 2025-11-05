import os
import socket
import logging
import threading
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import argparse
# import imutils # REMOVIDO: imutils.resize é substituído por cv2.resize
import cv2
# import dlib # REMOVIDO: Dlib não é mais necessário
import numpy as np
# from imutils import face_utils # REMOVIDO: face_utils não é mais necessário
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from flask import Flask

# --- NOVO: IMPORTS DO MEDIAPIPE ---
import mediapipe as mp

# --- NOVO: Constantes de Índices do MediaPipe ---
# Mapeamento dos 6 pontos do Dlib para os 468 do MediaPipe
MP_LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
# Pontos para o cálculo do MAR (Top, Bottom, Left, Right)
MP_MOUTH_INNER_INDICES = [13, 14, 78, 308]
# Pontos para desenhar o contorno da boca (opcional, mas mais bonito)
MP_MOUTH_OUTLINE_INDICES = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146]


# --- CONFIGURAÇÃO DO FLASK ---
app = Flask(__name__)
_status_lock = threading.Lock()
_status_fadiga = "Iniciando..."
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/status')
def get_status():
    with _status_lock:
        return _status_fadiga

def get_local_ip():
    # ... (função get_local_ip original, sem mudanças) ...
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('10.255.255.255', 1))
        IP = s.getsockname()[0]
    except Exception:
        IP = '127.0.0.1'
    finally:
        s.close()
    return IP

def rodar_servidor_flask(host='0.0.0.0', port=5000):
    ip_local = get_local_ip()
    print("[INFO] Servidor Flask rodando! Acesse no seu App Inventor:")
    print(f"[INFO] ==> http://{ip_local}:{port}/status")
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

# --- FUNÇÕES DE ASPECT RATIO ---
def calcular_ear(olho):
    # --- ESTA FUNÇÃO É 100% REUTILIZADA ---
    # O MediaPipe vai nos dar os 6 pontos exatos que o Dlib dava.
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])
    C = dist.euclidean(olho[0], olho[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

# --- REMOVIDO: calcular_mar(boca) ---
# A função original do Dlib (8 pontos) não é mais usada.
# O cálculo será feito inline com 4 pontos do MediaPipe.

# --- ARGPARSE ---
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="índice da webcam no sistema")
# --- REMOVIDO: --shape-predictor ---
# ap.add_argument("-p", "--shape-predictor", ...)
ap.add_argument("--no-display", action="store_true", help="não abrir janela (útil em servidor headless)")
args = vars(ap.parse_args())

# --- REMOVIDO: Lógica de verificação do predictor ---
# ... (toda a lógica de 'predictor_path' foi removida) ...


# --- CONFIGURAÇÃO DO FIREBASE ---
ref_firebase = None
try:
    # ... (lógica do Firebase original, sem mudanças) ...
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://safetruck-2f0a3-default-rtdb.firebaseio.com/'
    })
    ref_firebase = db.reference('status_caminhoneiro')
    print("[INFO] Conectado ao Firebase Realtime Database.")
except Exception as e:
    print(f"[ERRO] Não foi possível conectar ao Firebase: {e}")
    print("[WARN] Verifique se o 'serviceAccountKey.json' está na pasta correta.")
    print("[WARN] O script continuará sem a integração com Firebase.")


# --- CONSTANTES ---
LIMIAR_EAR = 0.25 # Mantenha por enquanto, mas talvez precise de ajuste
QTD_CONSEC_FRAMES_OLHOS = 20
# --- ATENÇÃO AQUI ---
LIMIAR_MAR = 0.4 # VALOR CHUTADO! O original (0.6) NÃO VAI FUNCIONAR.
# --- VOCÊ PRECISA AJUSTAR ESSE VALOR ---
QTD_CONSEC_FRAMES_BOCA = 15
LARGURA_FRAME = 700

# --- ESTADO ---
# ... (lógica de estado original, sem mudanças) ...
_contadores_lock = threading.Lock()
CONTADOR_OLHOS = 0
CONTADOR_BOCA = 0
TOTAL_BOCEJOS = 0
ALARME_ON = False
PONTOS_FADIGA = 0.0

# --- NOVO: Inicializa MediaPipe ---
print("[INFO] Carregando MediaPipe Face Mesh...")
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,                # Detecta apenas 1 rosto (igual ao Dlib)
    refine_landmarks=True,          # Importante: Habilita os 468 pontos (íris, lábios)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils # Para desenhar (opcional)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)


# --- REMOVIDO: Inicializa dlib ---
# detector = dlib.get_frontal_face_detector()
# preditor = dlib.shape_predictor(predictor_path)
# (inicio_esq, fim_esq) = ...
# (inicio_dir, fim_dir) = ...
# (inicio_boca, fim_boca) = ...

# Inicia thread de vídeo
print("[INFO] Iniciando fluxo de vídeo...")
# ... (lógica de inicialização da câmera original, sem mudanças) ...
vs = None
try:
    vs = VideoStream(src=args["webcam"]).start()
    time.sleep(1.0)
    frame_test = vs.read()
    if frame_test is None:
        raise Exception("VideoStream não retornou frame.")
except Exception as e:
    print(f"[WARN] VideoStream falhou ({e}), tentando cv2.VideoCapture fallback...")
    cap = cv2.VideoCapture(args["webcam"])
    if not cap.isOpened():
        raise SystemExit("[ERRO] Não foi possível abrir a câmera. Cheque o índice/permissões.")
    class CapWrapper:
        def __init__(self, cap):
            self.cap = cap
        def read(self):
            ret, frame = self.cap.read()
            return frame if ret else None
        def stop(self):
            self.cap.release()
    vs = CapWrapper(cap)


# Inicia servidor Flask em thread
# ... (lógica do Flask original, sem mudanças) ...
print("[INFO] Iniciando thread do servidor Flask...")
server_thread = threading.Thread(target=rodar_servidor_flask, daemon=True)
server_thread.start()

# --- FUNÇÃO FIREBASE ---
_last_data_sent_time = 0
_firebase_update_interval = 1.0 # Envia 1x por segundo

def enviar_dados_firebase(data):
    # ... (lógica do Firebase original, sem mudanças) ...
    global ref_firebase
    if not ref_firebase:
        return
    try:
        ref_firebase.set(data)
    except Exception as e:
        print(f"[WARN] Falha ao enviar dados para Firebase (thread): {e}")


# Configuração da janela
NOME_JANELA = "Detector de Fadiga (MediaPipe)"
# ... (lógica da janela, sem mudanças) ...
if not args.get("no_display"):
    try:
        cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
    except Exception:
        print("[WARN] Não foi possível criar janela (ambiente headless?). Use --no-display para rodar sem GUI.")


# --- Loop principal ---
try:
    while True:
        frame = vs.read()
        if frame is None:
            print("[AVISO] Não foi possível ler o frame da câmera. Tentando novamente...")
            time.sleep(0.5)
            continue

        # frame = imutils.resize(frame, width=LARGURA_FRAME) # Substituído
        frame = cv2.resize(frame, (LARGURA_FRAME, int(LARGURA_FRAME * frame.shape[0] / frame.shape[1])))
        
        # --- NOVO: LÓGICA MEDIAPIPE ---
        # 1. Converte BGR (OpenCV) para RGB (MediaPipe)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 2. Processa a imagem
        results = face_mesh.process(rgb_frame)

        texto_status_display = ""
        cor_status_display = (0, 255, 0)
        
        ear_frame = 0.0
        mar_frame = 0.0
        
        # 3. Verifica se um rosto foi detectado
        if not results.multi_face_landmarks:
            # --- LÓGICA "SEM ROSTO" (original, sem mudanças) ---
            with _contadores_lock:
                PONTOS_FADIGA = max(0.0, globals().get("PONTOS_FADIGA", 0.0) - 1.0)
                CONTADOR_OLHOS = 0
                CONTADOR_BOCA = 0
                ALARME_ON = False
            texto_status_display = "SEM ROSTO"
            with _status_lock:
                _status_fadiga = "Sem Rosto"
            cor_status_display = (100, 100, 100)
        else:
            # --- ROSTO DETECTADO ---
            # Pega o primeiro (e único) rosto
            face_landmarks = results.multi_face_landmarks[0]
            all_landmarks = face_landmarks.landmark
            
            # Pega as dimensões do frame para desnormalizar os pontos
            h, w, _ = frame.shape

            # --- NOVO: Extração de pontos para EAR ---
            # Constrói o array NumPy exatamente como o Dlib fazia,
            # mas usando os índices do MediaPipe.
            olho_esq_pts = np.array(
                [(all_landmarks[i].x * w, all_landmarks[i].y * h) for i in MP_LEFT_EYE_INDICES],
                dtype=np.int32
            )
            olho_dir_pts = np.array(
                [(all_landmarks[i].x * w, all_landmarks[i].y * h) for i in MP_RIGHT_EYE_INDICES],
                dtype=np.int32
            )
            
            # Calcula o EAR (a função 'calcular_ear' é a MESMA de antes)
            ear = (calcular_ear(olho_esq_pts) + calcular_ear(olho_dir_pts)) / 2.0

            # --- NOVO: Extração de pontos e cálculo do MAR ---
            # [13] = Lábio superior (centro)
            # [14] = Lábio inferior (centro)
            # [78] = Canto da boca (esquerda)
            # [308] = Canto da boca (direita)
            top_lip = all_landmarks[MP_MOUTH_INNER_INDICES[0]]
            bot_lip = all_landmarks[MP_MOUTH_INNER_INDICES[1]]
            l_lip = all_landmarks[MP_MOUTH_INNER_INDICES[2]]
            r_lip = all_landmarks[MP_MOUTH_INNER_INDICES[3]]

            # Calcula distâncias (usando coordenadas normalizadas, é mais rápido)
            dist_vertical = dist.euclidean((top_lip.x, top_lip.y), (bot_lip.x, bot_lip.y))
            dist_horizontal = dist.euclidean((l_lip.x, l_lip.y), (r_lip.x, r_lip.y))
            
            mar = dist_vertical / dist_horizontal if dist_horizontal != 0 else 0.0
            
            # Salva EAR/MAR do frame
            ear_frame = ear
            mar_frame = mar

            # --- NOVO: Desenho (Melhorado) ---
            # Desenha os contornos dos olhos (igual antes, mas com pontos do MP)
            cv2.drawContours(frame, [cv2.convexHull(olho_esq_pts)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(olho_dir_pts)], -1, (0, 255, 0), 1)
            
            # Desenha o contorno da boca (opcional)
            boca_desenho_pts = np.array(
                [(all_landmarks[i].x * w, all_landmarks[i].y * h) for i in MP_MOUTH_OUTLINE_INDICES],
                dtype=np.int32
            )
            cv2.drawContours(frame, [cv2.convexHull(boca_desenho_pts)], -1, (0, 255, 0), 1)


            # --- LÓGICA DE FADIGA (original, sem mudanças) ---
            # Esta é a sua "business logic". Ela não muda.
            # Ela apenas reage aos valores de 'ear' e 'mar'.
            with _contadores_lock:
                if mar > LIMIAR_MAR: # <-- LEMBRE-SE DE AJUSTAR O LIMIAR
                    CONTADOR_BOCA += 1
                else:
                    if CONTADOR_BOCA >= QTD_CONSEC_FRAMES_BOCA:
                        PONTOS_FADIGA = min(100.0, PONTOS_FADIGA + 15.0)
                        TOTAL_BOCEJOS += 1
                    CONTADOR_BOCA = 0

                if ear < LIMIAR_EAR: # <-- TALVEZ PRECISE DE AJUSTE
                    CONTADOR_OLHOS += 1
                    PONTOS_FADIGA = min(100.0, PONTOS_FADIGA + 1.0)
                    if CONTADOR_OLHOS >= QTD_CONSEC_FRAMES_OLHOS:
                        if not ALARME_ON:
                            ALARME_ON = True
                        PONTOS_FADIGA = 100.0
                else:
                    CONTADOR_OLHOS = 0
                    ALARME_ON = False
                    PONTOS_FADIGA = max(0.0, PONTOS_FADIGA - 0.5)

                PONTOS_FADIGA = max(0.0, min(100.0, PONTOS_FADIGA))

                if ALARME_ON:
                    status = "ALERTA CRITICO"
                    texto_status_display = "[ALERTA] SONOLENCIA!"
                    cor_status_display = (0, 0, 255)
                elif PONTOS_FADIGA > 70:
                    status = "FADIGA ALTA"
                    texto_status_display = "FADIGA ALTA"
                    cor_status_display = (0, 165, 255)
                elif PONTOS_FADIGA > 40:
                    status = "FADIGA MODERADA"
                    texto_status_display = "FADIGA MODERADA"
                    cor_status_display = (0, 255, 255)
                elif CONTADOR_BOCA > 1:
                    status = "BOCEJANDO"
                    texto_status_display = "BOCEJANDO"
                    cor_status_display = (255, 255, 0)
                else:
                    status = "NORMAL"
                    texto_status_display = "NORMAL"
                    cor_status_display = (0, 255, 0)

                with _status_lock:
                    _status_fadiga = status

            # ... (lógica cv2.putText original, sem mudanças) ...
            cv2.putText(frame, texto_status_display, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor_status_display, 2)
            cv2.putText(frame, f"Bocejos: {TOTAL_BOCEJOS}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"MAR: {mar:.2f}", (500, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(frame, f"EAR: {ear:.2f}", (500, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            texto_fadiga = f"NIVEL FADIGA: {PONTOS_FADIGA:.0f}%"
            cv2.putText(frame, texto_fadiga, (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if PONTOS_FADIGA<=40 else (0,165,255) if PONTOS_FADIGA<=70 else (0,0,255), 2)

        
        # --- LÓGICA DE ATUALIZAÇÃO FIREBASE (original, sem mudanças) ---
        agora = time.time()
        if ref_firebase and (agora - _last_data_sent_time > _firebase_update_interval):
            _last_data_sent_time = agora
            
            with _contadores_lock:
                current_pontos = PONTOS_FADIGA
                current_bocejos = TOTAL_BOCEJOS
                current_alarme = ALARME_ON
            with _status_lock:
                current_status = _status_fadiga

            data_to_send = {
                'timestamp': datetime.now().isoformat(),
                'status': current_status,
                'pontos_fadiga': round(current_pontos, 2),
                'total_bocejos': current_bocejos,
                'ear_atual': round(ear_frame, 2), # ear_frame e mar_frame agora são 0.0 se sem rosto
                'mar_atual': round(mar_frame, 2),
                'alarme_on': current_alarme
            }
            
            threading.Thread(target=enviar_dados_firebase, args=(data_to_send,), daemon=True).start()


        # mostra janela se não estiver em headless
        if not args.get("no_display"):
            try:
                cv2.imshow(NOME_JANELA, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except Exception:
                pass
except KeyboardInterrupt:
    print("[INFO] Interrompido pelo usuário.")
finally:
    # --- LÓGICA DE FINALIZAÇÃO (original, sem mudanças) ---
    print("[INFO] Finalizando...")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        vs.stop()
    except Exception:
        pass
    
    # --- NOVO: Limpa o MediaPipe ---
    try:
        face_mesh.close()
    except Exception:
        pass

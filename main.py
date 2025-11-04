# detector_fadiga_server.py (versão revisada)
import os
import socket
import logging
import threading
import time
from datetime import datetime

import argparse
import imutils
import cv2
import dlib
import numpy as np
from imutils import face_utils
from imutils.video import VideoStream
from scipy.spatial import distance as dist
from flask import Flask

# --- CONFIGURAÇÃO DO FLASK ---
app = Flask(__name__)
# variável de status protegida por lock
_status_lock = threading.Lock()
_status_fadiga = "Iniciando..."

# Silencia os logs "GET /status 200"
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

@app.route('/status')
def get_status():
    # retorna valor protegido por lock para evitar condição de corrida
    with _status_lock:
        return _status_fadiga

def get_local_ip():
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
    # use_reloader=False evita sub-processos indesejados quando rodando em thread
    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)

# --- FUNÇÕES DE ASPECT RATIO ---
def calcular_ear(olho):
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])
    C = dist.euclidean(olho[0], olho[3])
    ear = (A + B) / (2.0 * C) if C != 0 else 0.0
    return ear

def calcular_mar(boca):
    # pontos da boca interna (normalmente 8 pontos: 60-67)
    if len(boca) < 7:
        return 0.0
    A = dist.euclidean(boca[2], boca[6])
    B = dist.euclidean(boca[3], boca[5])
    C = dist.euclidean(boca[0], boca[4])
    mar = (A + B) / (2.0 * C) if C != 0 else 0.0
    return mar

# --- ARGPARSE (agora com mensagens amigáveis) ---
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0, help="índice da webcam no sistema")
ap.add_argument("-p", "--shape-predictor", required=False,
                help="path para o arquivo de marcos faciais (ex: shape_predictor_68_face_landmarks.dat)")
ap.add_argument("--no-display", action="store_true", help="não abrir janela (útil em servidor headless)")
args = vars(ap.parse_args())

# se não passou predictor, tenta procurar na pasta atual
predictor_path = args.get("shape_predictor")
if predictor_path is None:
    guessed = "shape_predictor_68_face_landmarks.dat"
    if os.path.exists(guessed):
        predictor_path = guessed
        print(f"[INFO] Usando preditor encontrado em: {predictor_path}")
    else:
        raise SystemExit("[ERRO] É necessário passar --shape-predictor ou colocar shape_predictor_68_face_landmarks.dat no diretório.")

if not os.path.exists(predictor_path):
    raise SystemExit(f"[ERRO] Arquivo não encontrado: {predictor_path}")

# --- CONSTANTES ---
LIMIAR_EAR = 0.25
QTD_CONSEC_FRAMES_OLHOS = 20
LIMIAR_MAR = 0.6
QTD_CONSEC_FRAMES_BOCA = 15
LARGURA_FRAME = 700

# --- ESTADO (protegido por lock quando necessário) ---
_contadores_lock = threading.Lock()
CONTADOR_OLHOS = 0
CONTADOR_BOCA = 0
TOTAL_BOCEJOS = 0
ALARME_ON = False
PONTOS_FADIGA = 0.0

# Inicializa dlib
print("[INFO] Carregando preditor de marcos faciais...")
detector = dlib.get_frontal_face_detector()
preditor = dlib.shape_predictor(predictor_path)

# índices dos marcos
(inicio_esq, fim_esq) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(inicio_dir, fim_dir) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# usa "inner_mouth" se disponível, senão "mouth"
if "inner_mouth" in face_utils.FACIAL_LANDMARKS_IDXS:
    (inicio_boca, fim_boca) = face_utils.FACIAL_LANDMARKS_IDXS["inner_mouth"]
else:
    (inicio_boca, fim_boca) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# Inicia thread de vídeo com fallback para VideoCapture
print("[INFO] Iniciando fluxo de vídeo...")
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
    # wrapper simples para compatibilidade com .read() do VideoStream
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
print("[INFO] Iniciando thread do servidor Flask...")
server_thread = threading.Thread(target=rodar_servidor_flask, daemon=True)
server_thread.start()

# Configuração da janela (se houver display)
NOME_JANELA = "Detector de Fadiga"
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
            # falha ao ler frame -> esperar e continuar
            print("[AVISO] Não foi possível ler o frame da câmera. Tentando novamente...")
            time.sleep(0.5)
            continue

        frame = imutils.resize(frame, width=LARGURA_FRAME)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)

        texto_status_display = ""
        cor_status_display = (0, 255, 0)

        # Se nenhum rosto detectado
        if not rects:
            with _contadores_lock:
                PONTOS_FADIGA = max(0.0, globals().get("PONTOS_FADIGA", 0.0) - 1.0)
                CONTADOR_OLHOS = 0
                CONTADOR_BOCA = 0
                ALARME_ON = False
            texto_status_display = "SEM ROSTO"
            with _status_lock:
                _status_fadiga = "Sem Rosto"
            cor_status_display = (100, 100, 100)

        for rect in rects:
            shape = preditor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            olho_esq = shape[inicio_esq:fim_esq]
            olho_dir = shape[inicio_dir:fim_dir]
            boca = shape[inicio_boca:fim_boca]

            ear = (calcular_ear(olho_esq) + calcular_ear(olho_dir)) / 2.0
            mar = calcular_mar(boca)

            # desenhos
            cv2.drawContours(frame, [cv2.convexHull(olho_esq)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(olho_dir)], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [cv2.convexHull(boca)], -1, (0, 255, 0), 1)

            with _contadores_lock:
                # bocejo
                if mar > LIMIAR_MAR:
                    CONTADOR_BOCA += 1
                else:
                    if CONTADOR_BOCA >= QTD_CONSEC_FRAMES_BOCA:
                        PONTOS_FADIGA = min(100.0, PONTOS_FADIGA + 15.0)
                        TOTAL_BOCEJOS += 1
                    CONTADOR_BOCA = 0

                # olhos
                if ear < LIMIAR_EAR:
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

                # garante limites
                PONTOS_FADIGA = max(0.0, min(100.0, PONTOS_FADIGA))

                # escolhe status
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

                # atualiza variável compartilhada
                with _status_lock:
                    _status_fadiga = status

            # textos na tela
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

        # mostra janela se não estiver em headless
        if not args.get("no_display"):
            try:
                cv2.imshow(NOME_JANELA, frame)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            except Exception:
                # em ambiente headless, apenas continue loop
                pass
except KeyboardInterrupt:
    print("[INFO] Interrompido pelo usuário.")
finally:
    print("[INFO] Finalizando...")
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    try:
        vs.stop()
    except Exception:
        # se wrapper cv2.VideoCapture, liberar já foi feito
        pass
        

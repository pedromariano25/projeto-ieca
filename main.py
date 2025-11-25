import os
import time
import threading
import logging
import argparse
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import mediapipe as mp
import serial
import serial.tools.list_ports # --- NOVO: Para listar portas disponíveis
from scipy.spatial import distance as dist
from flask import Flask
import firebase_admin
from firebase_admin import credentials, db

# --- CONFIGURAÇÕES DO SISTEMA ---
LIMIAR_EAR = 0.21            # Ponto de corte (Olho Fechado)
QTD_CONSEC_FRAMES_OLHOS = 45 # Tolerância (~2 segundos)
LIMIAR_MAR = 0.60            # Boca (Bocejo)
PORTA_ARDUINO = 'COM5'       # <--- SUA PORTA PADRÃO (Atualizada para COM5)
SKIP_FRAME_RATE = 2          # Otimização de FPS (Processa a cada 2 frames)

# --- CORES (INTERFACE CYBERPUNK) ---
C_CYAN   = (255, 255, 0)
C_VERDE  = (0, 255, 100)
C_AMARELO= (0, 200, 255)
C_LARANJA= (0, 100, 255)
C_VERM   = (0, 0, 255)
C_BRANCO = (240, 240, 240)
C_DARK   = (15, 15, 15)

# --- SETUP MEDIAPIPE (PONTOS DO ROSTO) ---
MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MP_MOUTH = [13, 14, 78, 308]
MP_NOSE = 1
MP_FACE_LEFT = 234
MP_FACE_RIGHT = 454

# --- FLASK (SERVIDOR LOCAL) ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
_status_lock = threading.Lock()
_status_fadiga = "BOOTING..."

@app.route('/status')
def get_status():
    with _status_lock: return _status_fadiga

def rodar_flask():
    try: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except: pass

# --- FUNÇÕES VISUAIS (HUD) ---
def draw_tech_bracket(img, x, y, w, h, cor, thickness=2, length=20):
    # Desenha cantoneiras tecnológicas
    # Top Left
    cv2.line(img, (x, y), (x + length, y), cor, thickness)
    cv2.line(img, (x, y), (x, y + length), cor, thickness)
    # Top Right
    cv2.line(img, (x + w, y), (x + w - length, y), cor, thickness)
    cv2.line(img, (x + w, y), (x + w, y + length), cor, thickness)
    # Bottom Left
    cv2.line(img, (x, y + h), (x + length, y + h), cor, thickness)
    cv2.line(img, (x, y + h), (x, y + h - length), cor, thickness)
    # Bottom Right
    cv2.line(img, (x + w, y + h), (x + w - length, y + h), cor, thickness)
    cv2.line(img, (x + w, y + h), (x + w, y + h - length), cor, thickness)

def draw_panel(img, x, y, w, h, alpha=0.7):
    # Fundo translúcido
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_DARK, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    # Borda sutil
    draw_tech_bracket(img, x, y, w, h, C_CYAN, 1, 10)

def draw_bar(img, x, y, w, h, pct, color):
    # Barra de progresso
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * (pct / 100.0))
    fill = max(0, min(fill, w))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    
def calcular_ear(olho):
    # Cálculo da razão de aspecto do olho (Eye Aspect Ratio)
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])
    C = dist.euclidean(olho[0], olho[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def verificar_pose(landmarks, w, h):
    # Verifica se o motorista está olhando para os lados
    nose = landmarks[MP_NOSE]
    left = landmarks[MP_FACE_LEFT]
    right = landmarks[MP_FACE_RIGHT]
    dl = dist.euclidean((nose.x, nose.y), (left.x, left.y))
    dr = dist.euclidean((nose.x, nose.y), (right.x, right.y))
    if dr == 0: dr = 0.001
    ratio = dl / dr
    if ratio < 0.35 or ratio > 2.8: return "VIRADO"
    return "FRENTE"

# --- MAIN ---
ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0)
ap.add_argument("--no-display", action="store_true")
args = vars(ap.parse_args())

# Conexões Externas (Firebase)
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://safetruck-2f0a3-default-rtdb.firebaseio.com/'})
    ref_fb = db.reference('status_caminhoneiro')
except: ref_fb = None

# Conexão Arduino (Serial) com DIAGNÓSTICO MELHORADO
arduino = None
try:
    arduino = serial.Serial(PORTA_ARDUINO, 9600, timeout=0.1)
    time.sleep(2) # Espera Arduino reiniciar após conexão
    print(f"[HW] Arduino CONECTADO na porta: {PORTA_ARDUINO}")
except Exception as e: 
    print(f"\n[ERRO] Não consegui conectar na {PORTA_ARDUINO}.")
    print(f"[DETALHE] {e}")
    print("\n--- LISTA DE PORTAS DISPONÍVEIS ---")
    # Lista todas as portas para ajudar você a achar a certa
    ports = serial.tools.list_ports.comports()
    if not ports:
        print(" > Nenhuma porta COM encontrada (Verifique o cabo USB!)")
    for port in ports:
        print(f" > {port.device}: {port.description}")
    print("-----------------------------------\n")
    print("[DICA] Se o Arduino apareceu em outra porta (ex: COM3), mude a linha 'PORTA_ARDUINO' no código.\n")

# Inicialização IA (MediaPipe)
mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
ear_queue = deque(maxlen=5)
mar_queue = deque(maxlen=5)
_lock = threading.Lock()

# Variáveis de Estado
CONT_OLHOS, CONT_BOCA, BOCEJOS = 0, 0, 0
ALARME, PONTOS = False, 0.0
frame_count = 0
scanner_line = 0
fps_start = time.time()
fps_val = 0

# Variáveis "Cache" (Para desenhar nos frames que a IA pula)
last_pose = "---"
last_ear = 0.0
last_points = [] 

# Inicia Flask em background
threading.Thread(target=rodar_flask, daemon=True).start()

# Configuração de Janela e Câmera
NOME_JANELA = "SAFETRUCK v4.0 PRO"
if not args.get("no_display"):
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap = cv2.VideoCapture(args["webcam"])
    # Tenta forçar resolução HD
    cap.set(3, 1280)
    cap.set(4, 720)
else:
    cap = cv2.VideoCapture(args["webcam"])

_last_fb = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        # Otimização visual e espelhamento
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        # --- OTIMIZAÇÃO: FRAME SKIPPING ---
        # Só roda a IA pesada a cada X frames
        frame_count += 1
        run_detection = (frame_count % SKIP_FRAME_RATE == 0)
        
        # Texto Padrão
        txt_main = "SISTEMA ATIVO"
        sub_txt = "MONITORANDO BIOMETRIA"
        cor_tema = C_CYAN
        
        if run_detection:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)
            
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                np_lm = lambda i: np.array([lm[i].x*w, lm[i].y*h], dtype=int)
                
                # Atualiza Cache
                last_pose = verificar_pose(lm, w, h)
                
                oe = np.array([np_lm(i) for i in MP_LEFT_EYE])
                od = np.array([np_lm(i) for i in MP_RIGHT_EYE])
                boca = np.array([np_lm(i) for i in MP_MOUTH])
                
                # Salva pontos para desenhar depois
                last_points = [oe, od]
                
                ear_now = (calcular_ear(oe) + calcular_ear(od)) / 2.0
                mar_now = dist.euclidean(boca[0], boca[1]) / (dist.euclidean(boca[2], boca[3]) + 0.001)
                
                ear_queue.append(ear_now)
                mar_queue.append(mar_now)
                
                # Usa a média suavizada (mais estável)
                ear = sum(ear_queue)/len(ear_queue)
                mar = sum(mar_queue)/len(mar_queue)
                last_ear = ear 

                with _lock:
                    if last_pose == "VIRADO":
                        txt_main = "ATENCAO"
                        sub_txt = "ROSTO VIRADO"
                        cor_tema = C_AMARELO
                        CONT_OLHOS = 0
                    else:
                        # Detecção de Bocejo
                        if mar > LIMIAR_MAR: CONT_BOCA += 1
                        else:
                            if CONT_BOCA > 15: 
                                PONTOS = min(100, PONTOS + 15)
                                BOCEJOS += 1
                            CONT_BOCA = 0
                        
                        # Detecção de Olhos Fechados
                        if ear < LIMIAR_EAR:
                            CONT_OLHOS += SKIP_FRAME_RATE # Compensa os frames pulados
                            PONTOS = min(100, PONTOS + 0.5)
                            if CONT_OLHOS >= QTD_CONSEC_FRAMES_OLHOS:
                                ALARME = True
                                PONTOS = 100
                        else:
                            CONT_OLHOS = 0
                            ALARME = False
                            PONTOS = max(0, PONTOS - 0.5)
        
        # --- DESENHO DA INTERFACE (RODA EM TODO FRAME) ---
        with _lock:
            if ALARME:
                txt_main = "PERIGO IMINENTE"
                sub_txt = "CONDUTOR DORMINDO"
                cor_tema = C_VERM
            elif PONTOS > 50:
                txt_main = "FADIGA ALTA"
                sub_txt = "RECOMENDA-SE PARADA"
                cor_tema = C_LARANJA
            elif last_pose == "VIRADO":
                txt_main = "DISTRACAO"
                sub_txt = "OLHE PARA FRENTE"
                cor_tema = C_AMARELO
            
            _status_fadiga = txt_main

        # Desenha os olhos e miras
        if len(last_points) > 0:
            cv2.polylines(frame, [cv2.convexHull(last_points[0])], True, cor_tema, 2, cv2.LINE_AA)
            cv2.polylines(frame, [cv2.convexHull(last_points[1])], True, cor_tema, 2, cv2.LINE_AA)
            # Crosshair
            cx_l = int(np.mean(last_points[0], axis=0)[0])
            cy_l = int(np.mean(last_points[0], axis=0)[1])
            cx_r = int(np.mean(last_points[1], axis=0)[0])
            cy_r = int(np.mean(last_points[1], axis=0)[1])
            l = 10
            cv2.line(frame, (cx_l-l, cy_l), (cx_l+l, cy_l), cor_tema, 1)
            cv2.line(frame, (cx_l, cy_l-l), (cx_l, cy_l+l), cor_tema, 1)
            cv2.line(frame, (cx_r-l, cy_r), (cx_r+l, cy_r), cor_tema, 1)
            cv2.line(frame, (cx_r, cy_r-l), (cx_r, cy_r+l), cor_tema, 1)

        # Efeito Scanner (Linha que desce)
        scanner_line += 10
        if scanner_line > h: scanner_line = 0
        line_color = C_VERM if ALARME else (0, 100, 0)
        cv2.line(frame, (0, scanner_line), (w, scanner_line), line_color, 1 if not ALARME else 4)

        # --- COMUNICAÇÃO COM ARDUINO ---
        # Envia '1' se ALARME for True, '0' se for False
        if arduino and frame_count % 5 == 0: # Limita envio para não travar Serial
            try: arduino.write(b'1' if ALARME else b'0')
            except: pass

        # --- HUD (ELEMENTOS GRAFICOS) ---
        if not args.get("no_display"):
            
            # Borda Vermelha de Pânico
            if ALARME:
                cv2.rectangle(frame, (0,0), (w,h), C_VERM, 30)

            # 1. Painel Esquerdo (Métricas)
            draw_panel(frame, 30, 30, 300, 300)
            
            cv2.putText(frame, "STATUS DO MOTORISTA", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BRANCO, 1, cv2.LINE_AA)
            cv2.line(frame, (45, 70), (315, 70), cor_tema, 2)
            
            draw_bar(frame, 45, 100, 270, 15, PONTOS, cor_tema)
            cv2.putText(frame, f"NIVEL FADIGA: {int(PONTOS)}%", (45, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_BRANCO, 1, cv2.LINE_AA)
            
            cv2.putText(frame, f"EAR (Olhos): {last_ear:.3f}", (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Timer Sono: {CONT_OLHOS}/{QTD_CONSEC_FRAMES_OLHOS}", (45, 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Bocejos: {BOCEJOS}", (45, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BRANCO, 1, cv2.LINE_AA)
            cv2.putText(frame, f"Pose: {last_pose}", (45, 225), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_AMARELO if last_pose == "VIRADO" else C_VERDE, 1, cv2.LINE_AA)

            # 2. Painel Central (Status)
            cw = 500
            cx = (w // 2) - (cw // 2)
            draw_panel(frame, cx, 30, cw, 100)
            
            t1_sz = cv2.getTextSize(txt_main, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
            t2_sz = cv2.getTextSize(sub_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            
            cv2.putText(frame, txt_main, (cx + (cw-t1_sz[0])//2, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, cor_tema, 2, cv2.LINE_AA)
            cv2.putText(frame, sub_txt, (cx + (cw-t2_sz[0])//2, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BRANCO, 1, cv2.LINE_AA)

            # 3. Rodapé (FPS + Hardware)
            draw_panel(frame, 30, h-70, w-60, 40)
            
            if frame_count % 10 == 0:
                fps_val = 10 / (time.time() - fps_start)
                fps_start = time.time()
            
            hora = datetime.now().strftime("%H:%M:%S")
            hw_txt = "ARDUINO ONLINE" if arduino else "ARDUINO OFFLINE"
            hw_col = C_VERDE if arduino else (100,100,100)

            cv2.putText(frame, f"FPS: {int(fps_val)}", (50, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 1)
            cv2.putText(frame, f"TIME: {hora}", (200, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BRANCO, 1)
            cv2.putText(frame, hw_txt, (w-250, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, hw_col, 1)

            cv2.imshow(NOME_JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

        # Envio Firebase (Limitado a 1x por seg)
        if ref_fb and (time.time() - _last_fb > 1.0):
            _last_fb = time.time()
            dt = {'status': txt_main, 'pontos': round(PONTOS,2), 'alarme': ALARME}
            threading.Thread(target=lambda: ref_fb.set(dt), daemon=True).start()

except KeyboardInterrupt: pass
finally:
    cv2.destroyAllWindows()
    cap.release()
    if arduino: arduino.close()
    print("\n[INFO] Sistema encerrado.")

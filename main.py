import os
import time
import threading
import logging
import argparse
from collections import deque
from datetime import datetime
import winsound # Som do Windows
import pyttsx3  # Voz Robótica

import cv2
import numpy as np
import mediapipe as mp
import serial
import serial.tools.list_ports
from scipy.spatial import distance as dist
from flask import Flask
import firebase_admin
from firebase_admin import credentials, db

# --- CONFIGURAÇÕES DO SISTEMA ---
LIMIAR_EAR = 0.21            # Olho Fechado (Se usar óculos, tente 0.19)
QTD_CONSEC_FRAMES_OLHOS = 45 # Tempo até apitar (~2 seg)
LIMIAR_MAR = 0.70            # Boca Aberta (Aumentei um pouco pra não dar falso positivo)
PORTA_ARDUINO = 'COM5'       # <--- CONFIRA SUA PORTA
SKIP_FRAME_RATE = 2          # Otimização

# --- CORES CYBERPUNK ---
C_CYAN    = (255, 255, 0)
C_VERDE   = (0, 255, 100)
C_AMARELO = (0, 200, 255)
C_LARANJA = (0, 100, 255)
C_VERM    = (0, 0, 255)
C_BRANCO  = (240, 240, 240)
C_DARK    = (15, 15, 15)

# --- SETUP MEDIAPIPE ---
MP_LEFT_EYE = [33, 160, 158, 133, 153, 144]
MP_RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MP_MOUTH = [13, 14, 78, 308]
MP_NOSE = 1
MP_FACE_LEFT = 234
MP_FACE_RIGHT = 454

# --- SISTEMA DE SOM (AMBULÂNCIA + VOZ) ---
def tocar_sirene_avancada():
    def _tocar():
        try:
            # 1. Sirene (Uóóó Uóóó)
            for _ in range(3):
                winsound.Beep(700, 300)
                winsound.Beep(1800, 300)
            
            # 2. Voz
            engine = pyttsx3.init()
            engine.setProperty('rate', 230)
            engine.say("PERIGO! ACORDA MOTORISTA!")
            engine.runAndWait()
        except: pass
    
    threading.Thread(target=_tocar, daemon=True).start()

# --- FLASK ---
app = Flask(__name__)
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)
_status_lock = threading.Lock()
_status_fadiga = "INICIANDO..."

@app.route('/status')
def get_status():
    with _status_lock: return _status_fadiga

def rodar_flask():
    try: app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
    except: pass

# --- FUNÇÕES VISUAIS (HUD QUE VOCÊ GOSTOU) ---
def draw_tech_bracket(img, x, y, w, h, cor, thickness=2, length=20):
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
    overlay = img.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), C_DARK, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
    draw_tech_bracket(img, x, y, w, h, C_CYAN, 1, 10)

def draw_bar(img, x, y, w, h, pct, color):
    cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
    fill = int(w * (pct / 100.0))
    fill = max(0, min(fill, w))
    cv2.rectangle(img, (x, y), (x + fill, y + h), color, -1)
    
def calcular_ear(olho):
    A = dist.euclidean(olho[1], olho[5])
    B = dist.euclidean(olho[2], olho[4])
    C = dist.euclidean(olho[0], olho[3])
    return (A + B) / (2.0 * C) if C != 0 else 0.0

def verificar_pose(landmarks, w, h):
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

# Firebase
try:
    cred = credentials.Certificate('serviceAccountKey.json')
    firebase_admin.initialize_app(cred, {'databaseURL': 'https://safetruck-2f0a3-default-rtdb.firebaseio.com/'})
    ref_fb = db.reference('status_caminhoneiro')
except: ref_fb = None

# Arduino
arduino = None
try:
    arduino = serial.Serial(PORTA_ARDUINO, 9600, timeout=0.1)
    time.sleep(2) 
    print(f"[HW] Arduino CONECTADO: {PORTA_ARDUINO}")
except Exception as e: 
    print(f"\n[ERRO] Arduino nao encontrado. Rodando sem ele.")

mp_face = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True)
ear_queue = deque(maxlen=5)
mar_queue = deque(maxlen=5)
_lock = threading.Lock()

CONT_OLHOS, CONT_BOCA, BOCEJOS = 0, 0, 0
ALARME, PONTOS = False, 0.0
frame_count = 0
scanner_line = 0
fps_start = time.time()
fps_val = 0 # <--- AQUI ESTAVA O ERRO, AGORA ESTÁ CORRIGIDO
_last_sirene = 0
_last_fb = 0

last_pose = "---"
last_ear = 0.0
last_points = [] 

threading.Thread(target=rodar_flask, daemon=True).start()

NOME_JANELA = "SAFETRUCK v5.0 ULTIMATE"
if not args.get("no_display"):
    cv2.namedWindow(NOME_JANELA, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(NOME_JANELA, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cap = cv2.VideoCapture(args["webcam"])
    cap.set(3, 1280)
    cap.set(4, 720)
else:
    cap = cv2.VideoCapture(args["webcam"])

try:
    while True:
        ret, frame = cap.read()
        if not ret: continue
        
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        
        frame_count += 1
        run_detection = (frame_count % SKIP_FRAME_RATE == 0)
        
        txt_main = "SISTEMA ATIVO"
        sub_txt = "MONITORANDO BIOMETRIA"
        cor_tema = C_CYAN
        
        if run_detection:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = mp_face.process(rgb)
            
            if res.multi_face_landmarks:
                lm = res.multi_face_landmarks[0].landmark
                np_lm = lambda i: np.array([lm[i].x*w, lm[i].y*h], dtype=int)
                
                last_pose = verificar_pose(lm, w, h)
                
                oe = np.array([np_lm(i) for i in MP_LEFT_EYE])
                od = np.array([np_lm(i) for i in MP_RIGHT_EYE])
                boca = np.array([np_lm(i) for i in MP_MOUTH])
                
                last_points = [oe, od]
                
                ear_now = (calcular_ear(oe) + calcular_ear(od)) / 2.0
                mar_now = dist.euclidean(boca[0], boca[1]) / (dist.euclidean(boca[2], boca[3]) + 0.001)
                
                ear_queue.append(ear_now)
                mar_queue.append(mar_now)
                
                ear = sum(ear_queue)/len(ear_queue)
                mar = sum(mar_queue)/len(mar_queue)
                last_ear = ear 

                with _lock:
                    # 1. Checa se virou o rosto
                    if last_pose == "VIRADO":
                        txt_main = "ATENCAO"
                        sub_txt = "ROSTO VIRADO"
                        cor_tema = C_AMARELO
                        CONT_OLHOS = 0
                    else:
                        # 2. Checa Bocejo
                        if mar > LIMIAR_MAR: 
                            CONT_BOCA += 1
                        else:
                            if CONT_BOCA > 10: # Se ficou boca aberta tempo suficiente
                                PONTOS = min(100, PONTOS + 10)
                                BOCEJOS += 1
                            CONT_BOCA = 0
                        
                        # 3. Checa Olhos (Sono)
                        if ear < LIMIAR_EAR:
                            CONT_OLHOS += SKIP_FRAME_RATE
                            PONTOS = min(100, PONTOS + 0.5)
                            if CONT_OLHOS >= QTD_CONSEC_FRAMES_OLHOS:
                                ALARME = True
                                PONTOS = 100
                        else:
                            CONT_OLHOS = 0
                            ALARME = False
                            PONTOS = max(0, PONTOS - 0.5)
        
        # --- LÓGICA DE ALARME E ARDUINO ---
        with _lock:
            if ALARME:
                txt_main = "PERIGO IMINENTE"
                sub_txt = "CONDUTOR DORMINDO"
                cor_tema = C_VERM
                
                # SOM: Toca sirene a cada 5 segundos
                if time.time() - _last_sirene > 5.0:
                    tocar_sirene_avancada()
                    _last_sirene = time.time()
                    BOCEJOS += 1 # Conta como evento critico

            elif PONTOS > 50:
                txt_main = "FADIGA ALTA"
                sub_txt = "RECOMENDA-SE PARADA"
                cor_tema = C_LARANJA
            
            _status_fadiga = txt_main

        # Arduino (Corrigido com \n)
        if arduino and frame_count % 5 == 0:
            try:
                if ALARME: arduino.write(b'H\n')
                else: arduino.write(b'L\n')
            except: pass
            
        # Firebase (Completo)
        if ref_fb and (time.time() - _last_fb > 1.0):
            _last_fb = time.time()
            data = {
                'status': txt_main, 
                'fadiga': int(PONTOS), 
                'alarme': bool(ALARME),
                'bocejos': int(BOCEJOS),
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            threading.Thread(target=lambda: ref_fb.set(data), daemon=True).start()

        # --- DESENHO HUD (INTERFACE LIMPA) ---
        if not args.get("no_display"):
            if len(last_points) > 0:
                cv2.polylines(frame, [cv2.convexHull(last_points[0])], True, cor_tema, 2, cv2.LINE_AA)
                cv2.polylines(frame, [cv2.convexHull(last_points[1])], True, cor_tema, 2, cv2.LINE_AA)

            # Scanner
            scanner_line += 10
            if scanner_line > h: scanner_line = 0
            line_color = C_VERM if ALARME else (0, 100, 0)
            cv2.line(frame, (0, scanner_line), (w, scanner_line), line_color, 1 if not ALARME else 4)

            if ALARME: cv2.rectangle(frame, (0,0), (w,h), C_VERM, 30)

            # Painel Esquerdo
            draw_panel(frame, 30, 30, 300, 300)
            cv2.putText(frame, "STATUS DO MOTORISTA", (45, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BRANCO, 1)
            cv2.line(frame, (45, 70), (315, 70), cor_tema, 2)
            draw_bar(frame, 45, 100, 270, 15, PONTOS, cor_tema)
            cv2.putText(frame, f"FADIGA: {int(PONTOS)}%", (45, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.4, C_BRANCO, 1)
            cv2.putText(frame, f"EAR: {last_ear:.3f}", (45, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_CYAN, 1)
            cv2.putText(frame, f"Bocejos: {BOCEJOS}", (45, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, C_BRANCO, 1)

            # Painel Central
            cw = 500
            cx = (w // 2) - (cw // 2)
            draw_panel(frame, cx, 30, cw, 100)
            t1_sz = cv2.getTextSize(txt_main, cv2.FONT_HERSHEY_DUPLEX, 1.0, 2)[0]
            t2_sz = cv2.getTextSize(sub_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(frame, txt_main, (cx + (cw-t1_sz[0])//2, 80), cv2.FONT_HERSHEY_DUPLEX, 1.0, cor_tema, 2)
            cv2.putText(frame, sub_txt, (cx + (cw-t2_sz[0])//2, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_BRANCO, 1)

            # Rodapé
            draw_panel(frame, 30, h-70, w-60, 40)
            if frame_count % 10 == 0:
                fps_val = 10 / (time.time() - fps_start)
                fps_start = time.time()
            
            hw_txt = "ARDUINO ON" if arduino else "ARDUINO OFF"
            cv2.putText(frame, f"FPS: {int(fps_val)}", (50, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_CYAN, 1)
            cv2.putText(frame, hw_txt, (w-250, h-45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, C_VERDE if arduino else (100,100,100), 1)

            cv2.imshow(NOME_JANELA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'): break

except KeyboardInterrupt: pass
finally:
    cv2.destroyAllWindows()
    cap.release()
    if arduino: arduino.close()
    print("\n[INFO] Sistema encerrado.")

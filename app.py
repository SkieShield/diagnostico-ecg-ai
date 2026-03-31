import streamlit as st
import tensorflow as tf
import neurokit2 as nk
import pandas as pd
import numpy as np
import os
import cv2
import base64
from PIL import Image
from datetime import datetime

# --- CONFIGURAÇÃO MASTER SENTINEL v15.0: VISION ELITE ---
st.set_page_config(page_title="CardioAI Master: Sentinel v15.0", page_icon="🩺", layout="wide")

# Estética Master (Dark Mode & Vision Aura)
st.markdown("""
<style>
    .main { background-color: #0b121f; color: #ffffff; }
    .report-box { background: rgba(0, 180, 216, 0.05); padding: 25px; border-radius: 12px; border: 1px solid #00b4d8; }
    .stMetric { background: rgba(255, 255, 255, 0.03); padding: 15px; border-radius: 10px; }
    .vision-tag { background: #00b4d8; color: white; padding: 2px 8px; border-radius: 5px; font-size: 0.8em; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# 1. Carregamento do Motor Master v15.0
@st.cache_resource
def carregar_ia():
    file_name = 'modelo_ecg_elite.h5'
    if os.path.exists(file_name):
        return tf.keras.models.load_model(file_name)
    return None

model = carregar_ia()

# --- MÓDULO DIGITALIZADOR SENTINEL v15.0 (VISÃO COMPUTACIONAL) ---
def digitalizar_ecg(image_bytes):
    # Converter bytes em imagem OpenCV
    nparr = np.frombuffer(image_bytes.read(), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Surgical Grid Removal (v15.0 Improved)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 25))
    grid_mask = cv2.add(cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel), 
                        cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel))
    
    # Reconstrução do Sinal Digital
    clean_img = cv2.inpaint(gray, grid_mask, 3, cv2.INPAINT_TELEA)
    _, final_thresh = cv2.threshold(clean_img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Extração de Coordenadas (Coluna por Coluna)
    points = []
    h, w = final_thresh.shape
    for x in range(w):
        col = final_thresh[:, x]
        indices = np.where(col == 255)[0]
        if len(indices) > 0:
            points.append(h - np.mean(indices)) # Inverter Y (0 é topo)
        else:
            points.append(points[-1] if points else h/2)
    
    return np.array(points), img

# --- INTERFACE PRINCIPAL ---
st.title("🩺 master v15.0: Sentinel Clinical Vision")
st.write("Diagnosticador Bioelétrico com Módulo de Digitalização de Imagem Integrado (OCR-ECG)")
st.markdown("---")

col_left, col_right = st.columns([1, 2])

with col_left:
    st.header("📤 Entrada Master")
    # Suporte Híbrido: Sinal Digital ou Foto de Papel
    file = st.file_uploader("Upload: .csv, .txt, .dat, .jpg, .png", type=["csv", "txt", "dat", "jpg", "png", "jpeg"])
    
    if file:
        is_image = file.name.lower().endswith(('.jpg', '.png', '.jpeg'))
        
        if is_image:
            st.info("🖼️ **Vision Sentinel:** Detectamos uma imagem. Digitalizando ECG...")
            sinal_extraido, img_original = digitalizar_ecg(file)
            st.image(img_original, caption="Foto Original do Exame", use_container_width=True)
            sinal_final = sinal_extraido
        else:
            df = pd.read_csv(file)
            sinal_final = df.iloc[:, 0].values
            st.success("✅ Sinal Digital Sincronizado.")

        if st.button("🚀 INICIAR DIAGNÓSTICO MASTER"):
            with st.spinner("Analisando Bioeletricidade Sentinel..."):
                # A. Limpeza Neural (Neurokit2)
                cleaned = nk.ecg_clean(sinal_final, sampling_rate=360)
                
                # B. Predição Master (ResNet/CNN)
                if model is not None:
                    target_len = model.input_shape[1]
                    target_ch = model.input_shape[2]
                    
                    input_ia = cleaned[:target_len]
                    if len(input_ia) < target_len: input_ia = np.pad(input_ia, (0, target_len - len(input_ia)))
                    input_ia = (input_ia - np.mean(input_ia)) / (np.std(input_ia) + 1e-8)
                    
                    # Ajuste de Canais (1 ou 12)
                    if target_ch == 12:
                        input_ia = np.tile(input_ia.reshape(target_len, 1), (1, 12)).reshape(1, target_len, 12)
                    else:
                        input_ia = input_ia.reshape(1, target_len, 1)

                    res = model.predict(input_ia, verbose=0)[0]
                    classe_id = np.argmax(res)
                    conf = np.max(res) * 100
                    
                    veredito = "NORMAL (RITMO SINUSAL)" if classe_id == 0 else "ANORMALIA DETECTADA (ARRITMIA)"
                    
                    with col_right:
                        st.header("📊 Veredito Sentinel v15.0")
                        
                        # --- LAUDO PROFISSIONAL CORE ---
                        st.markdown(f"""
                        <div class="report-box">
                            <h2 style='color: #00b4d8; text-align: center;'>LAUDO DE DIAGNÓSTICO MASTER</h2>
                            <p style='text-align: right;'><b>ID:</b> SEC-{np.random.randint(100, 999)} | {datetime.now().strftime('%d/%m/%Y %H:%M')}</p>
                            <hr style='border: 1px solid rgba(0, 180, 216, 0.3);'>
                            <p><b>Fonte dos Dados:</b> {"FOTO DE PAPEL EXTRASÍSTOLE" if is_image else "SINAL DIGITAL PURO"}</p>
                            <div style='background: rgba(0, 180, 216, 0.1); padding: 15px; border-radius: 8px;'>
                                <h1 style='color: #ffffff; margin: 0;'>{veredito}</h1>
                                <p style='color: #00b4d8;'>Confiança Neural: {conf:.2f}%</p>
                            </div>
                            <p style='margin-top: 15px;'><i>O motor ResNet-1D Sentinel v15.0 analisou as derivações bioelétricas extraídas com precisão de elite.</i></p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.line_chart(cleaned[:1000], height=250)
                        
                        # Download do Laudo
                        laudo_txt = f"LAUDO CARDIOAI MASTER v15.0\nData: {datetime.now()}\nDiagnostico: {veredito}\nConfianca: {conf:.2f}%"
                        b64 = base64.b64encode(laudo_txt.encode()).decode()
                        st.markdown(f'<a href="data:file/txt;base64,{b64}" download="laudo_master_v150.txt" style="display: block; text-align: center; background: #00b4d8; color: white; padding: 10px; border-radius: 5px; text-decoration: none; font-weight: bold; margin-top: 20px;">📥 DOWNLOAD LAUDO CLÍNICO</a>', unsafe_allow_html=True)

if not file:
    with col_right:
        st.info("💡 **Dica Master:** Agora você pode fazer o upload da foto do ECG em papel milimetrado. O sistema removerá a grade e digitalizará o sinal automaticamente!")
        st.write("Suporte: Sinais digitais (.csv, .dat) ou Imagens Físicas Escaneadas (.jpg, .png).")

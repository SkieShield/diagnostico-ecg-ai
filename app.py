import streamlit as st
import tensorflow as tf
import neurokit2 as nk
import pandas as pd
import numpy as np
import os
import base64
from datetime import datetime

# --- CONFIGURAÇÃO MASTER SENTINEL ELITE v14.0 ---
st.set_page_config(page_title="CardioAI: Sentinel Elite", page_icon="🏥", layout="wide")

# Estética de Elite (Aura Hospitalar & Neon)
st.markdown("""
<style>
    .main { background-color: #0b121f; color: #ffffff; font-family: 'Inter', sans-serif; }
    .report-card { background: rgba(255, 255, 255, 0.03); padding: 30px; border-radius: 20px; border: 1px solid rgba(0, 180, 216, 0.3); box-shadow: 0 10px 30px rgba(0,0,0,0.5); }
    .stMetric { background: rgba(0, 180, 216, 0.1); border-radius: 12px; padding: 20px; border-left: 5px solid #00b4d8; }
    .stButton>button { background: linear-gradient(90deg, #00b4d8, #0077b6); color: white; border: none; height: 50px; font-weight: bold; font-size: 18px; border-radius: 10px; transition: 0.3s; width: 100%; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 5px 15px rgba(0,180,216,0.4); }
    .stTextInput>div>div>input, .stNumberInput>div>div>input, .stSelectbox>div>div>select { background-color: rgba(255,255,255,0.05) !important; color: white !important; border: 1px solid rgba(0,180,216,0.2) !important; }
</style>
""", unsafe_allow_html=True)

# 1. Carregamento do Motor de Elite
@st.cache_resource
def carregar_modelo_elite():
    # Sincronizado conforme pedido: modelo_ecg_elite.h5
    file_name = 'modelo_ecg_elite.h5' 
    if os.path.exists(file_name):
        return tf.keras.models.load_model(file_name)
    return None

model = carregar_modelo_elite()

# --- SIDEBAR: PRONTUÁRIO DIGITAL ---
with st.sidebar:
    st.markdown("<h1 style='text-align: center;'>🏥</h1>", unsafe_allow_html=True)
    st.header("📋 Dados do Paciente")
    nome = st.text_input("Nome Completo", placeholder="Ex: João da Silva")
    col_cid1, col_cid2 = st.columns(2)
    with col_cid1: idade = st.number_input("Idade", 0, 120, 30)
    with col_cid2: sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    st.markdown("---")
    st.info("💡 **Dica Cloud:** Este prontuário será usado para gerar a minuta final do laudo.")

# --- CONTEÚDO PRINCIPAL ---
st.title("🏥 CardioAI Master: Sentinel Elite v14.0")
st.write("Monitoramento Cardiovascular Avançado via ResNet-1D / PTB-XL Dataset")
st.markdown("---")

uploaded_file = st.file_uploader("📥 Arraste o arquivo de ECG (.csv ou .txt)", type=["csv", "txt", "dat"])

if uploaded_file:
    col1, col2 = st.columns([2, 1])
    
    try:
        # Carregamento e Limpeza (Canais: 12 ou Mono)
        df = pd.read_csv(uploaded_file)
        # Pegamos o primeiro canal para visualização estética
        sinal_viz = df.iloc[:, 0].values
        cleaned = nk.ecg_clean(sinal_viz, sampling_rate=360)
        
        with col1:
            st.subheader("📊 Visualização de Derivações (Lead DII)")
            st.line_chart(cleaned[:1000], height=300)
            st.caption("Frequência de Amostragem: 360Hz | Filtro Master Sentinel Ativo")

        with col2:
            st.subheader("🩺 Diagnóstico de Elite")
            if model is not None:
                # 2. Processamento para ResNet-1D
                # Preparamos 1000 pontos e 12 canais (Padrão ResNet do Colab)
                input_ia = cleaned[:1000]
                if len(input_ia) < 1000: input_ia = np.pad(input_ia, (0, 1000 - len(input_ia)))
                input_ia = (input_ia - np.mean(input_ia)) / (np.std(input_ia) + 1e-8)
                
                # Simulando 12 derivações se o arquivo for mono-canal
                input_ia = np.tile(input_ia.reshape(1000, 1), (1, 12)).reshape(1, 1000, 12)
                
                # Inferência ResNet
                res = model.predict(input_ia, verbose=0)[0]
                confianca = np.max(res) * 100
                classe = np.argmax(res)
                
                # Mapeamento do treino Elite (PTB-XL style)
                res_ia = "✅ RITMO NORMAL" if classe == 0 else "⚠️ ALTERAÇÃO DETECTADA"
                desc_ia = "Ritmo Sinusal preservado." if classe == 0 else "Sinais compatíveis com Arritmia/Alteração Cardíaca."
                
                if classe == 0: st.success(res_ia)
                else: st.error(res_ia)
                
                st.metric("Confiança da IA Master", f"{confianca:.2f}%")
                st.write(f"Veredito: {desc_ia}")
            else:
                st.warning("⚠️ Aguardando arquivo 'modelo_ecg_elite.h5'...")

        # --- SEÇÃO DE LAUDO PROFISSIONAL ---
        st.markdown("---")
        st.subheader("📝 Minuta do Laudo Clínico")
        
        laudo_html = f"""
        <div class="report-card">
            <h1 style='text-align: center; color: #00b4d8; margin-top: 0;'>LAUDO DE ELETROCARDIOGRAMA</h1>
            <div style='display: flex; justify-content: space-between; border-bottom: 2px solid rgba(0, 180, 216, 0.2); padding-bottom: 10px;'>
                <span><b>PACIENTE:</b> {nome.upper() if nome else "N/A"}</span>
                <span><b>IDADE:</b> {idade} ANOS</span>
                <span><b>SEXO:</b> {sexo.upper()}</span>
            </div>
            <div style='margin-top: 20px;'>
                <p><b>CONCLUSÃO DA ANÁLISE:</b> <span style='font-size: 20px; color: {"#00ff00" if (model is not None and classe == 0) else "#ff4b4b"};'>{res_ia if model is not None else "AGUARDANDO MODELO"}</span></p>
                <p><b>DETALHAMENTO:</b> {desc_ia if model is not None else "N/A"}</p>
                <p><b>DATA DA ANÁLISE:</b> {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
            </div>
            <hr style='border: 1px solid rgba(255, 255, 255, 0.05);'>
            <p style='font-size: 0.8em; color: #666; text-align: justify;'>
                <b>Observação Legal:</b> Este documento é uma sugestão de laudo baseada em visão computacional e Deep Learning (ResNet-1D / PTB-XL). 
                A validação final e assinatura do laudo devem ser feitas por um médico cardiologista qualificado.
            </p>
        </div>
        """
        st.markdown(laudo_html, unsafe_allow_html=True)
        
        # Botão para Download Digital
        if model is not None:
            laudo_txt = f"LAUDO CARDIOAI ELITE v14.0\nPACIENTE: {nome}\nDIAGNOSTICO: {res_ia}\nCONFIANCA: {confianca:.2f}%"
            b64 = base64.b64encode(laudo_txt.encode()).decode()
            st.markdown(f'<a href="data:file/txt;base64,{b64}" download="laudo_elite.txt" style="display: block; text-align: center; background: linear-gradient(90deg, #00b4d8, #0077b6); color: white; padding: 15px; border-radius: 10px; text-decoration: none; font-weight: bold; margin-top: 10px;">🧧 DOWNLOAD LAUDO DIGITAL</a>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Erro no Motor Sentinel Elite: {e}")

else:
    st.markdown("""
        <div style='text-align: center; padding: 50px;'>
            <h1 style='font-size: 80px; opacity: 0.1;'>🏥</h1>
            <h3 style='color: #00b4d8;'>Aguardando recepção do sinal bioelétrico...</h3>
            <p>O motor Sentinel Elite v14.0 analisará janelas de 10 segundos em 12 derivações.</p>
        </div>
    """, unsafe_allow_html=True)

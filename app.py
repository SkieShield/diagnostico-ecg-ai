import streamlit as st
import tensorflow as tf
import numpy as np
import neurokit2 as nk
import pandas as pd
import os

# --- Configurações CardioAI Master v12.7 ---
st.set_page_config(page_title="CardioAI Master Console", page_icon="🩺", layout="wide")

st.markdown("""
<style>
    .main { background-color: #0b121f; color: #ffffff; }
    .stButton>button { background-color: #00b4d8; color: white; border-radius: 8px; border: none; font-weight: bold; width: 100%; height: 50px; transition: 0.3s; }
    .stButton>button:hover { background-color: #0096c7; transform: scale(1.02); }
    .stAlert { border-radius: 12px; background: rgba(0, 180, 216, 0.1); border: 1px solid #00b4d8; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 CardioAI Master: Análise de ECG com IA")
st.write("Faça o upload do seu sinal de ECG (formato .csv ou .txt) para diagnóstico automático via Stanford EchoNext v12.7.")

# 1. Carregar o Modelo Treinado v12.7
@st.cache_resource
def load_my_model():
    model_name = 'meu_modelo_ecg.h5'
    if os.path.exists(model_name):
        return tf.keras.models.load_model(model_name)
    else:
        st.error(f"❌ Erro Clínica: Arquivo {model_name} não encontrado na raiz.")
        return None

model = load_my_model()

# 2. Upload do arquivo pelo usuário
uploaded_file = st.file_uploader("Escolha o arquivo de ECG", type=["csv", "txt"])

if uploaded_file is not None:
    try:
        # Ler os dados (Primeira Coluna = Sinal)
        df = pd.read_csv(uploaded_file)
        sinal = df.iloc[:, 0].values
        
        # 3. Processamento Automático (Limpeza High-Fidelity v12.7)
        with st.spinner("Realizando limpeza cirúrgica de sinal (NeuroKit2)..."):
            sinal_clean = nk.ecg_clean(sinal, sampling_rate=360)
            st.subheader("Visualização Clínica (Clean Trace)")
            st.line_chart(sinal_clean[:1000]) 
            
# 4. Predição da IA Stanford v12.7 (Loop de Análise Global)
        if model is not None:
            with st.spinner("Analisando todos os batimentos detectados..."):
                # A. Detecção de Picos R (Master v12.7)
                try:
                    picos, _ = nk.ecg_peaks(sinal_clean, sampling_rate=360)
                    indices = picos["ECG_R_Peaks"][picos["ECG_R_Peaks"] == 1].index.values
                    st.info(f"🔍 Detectamos {len(indices)} batimentos no sinal total.")
                    
                    if len(indices) == 0:
                        st.error("❌ Erro: Nenhum batimento rítmico detectado no sinal.")
                        st.stop()

                    all_preds = []
                    # B. Loop de Inferência em Massa
                    for idx in indices:
                        start, end = idx - 187//2, idx + 187//2
                        if start > 0 and end < len(sinal_clean):
                            beat = sinal_clean[start:end]
                            if len(beat) == 187:
                                beat = (beat - np.mean(beat)) / (np.std(beat) + 1e-8)
                                beat = beat.reshape(1, 187, 1)
                                all_preds.append(model.predict(beat, verbose=0)[0])
                    
                    # C. Agregação Diagnóstica Master
                    if all_preds:
                        final_pred = np.mean(all_preds, axis=0) # Média Probabilística
                        classe_id = np.argmax(final_pred)
                        prob = np.max(final_pred) * 100
                        
                        diagnosticos = ["Normal (Ritmo Sinusal)", "Arritmia Supraventricular", "Batimento Ventricular Prematuro (PVC)", "Fusão de Ritmo", "Desconhecido/Inconclusivo"]
                        
                        st.success(f"🚀 DIAGNÓSTICO GLOBAL: {diagnosticos[classe_id]}")
                        st.metric("Confiança Média do Ciclo", f"{prob:.2f}%")
                        
                        # Alertas de Anomalia Individual (Flag de Risco)
                        max_classes = [np.argmax(p) for p in all_preds]
                        if 1 in max_classes or 2 in max_classes:
                            st.warning("⚠️ **ALERTA CLINICO:** Detectamos focos isolados de Arritmia/Extrasístole durante o monitoramento.")
                        
                except Exception as ex:
                    st.error(f"Erro na análise rítmica: {ex}")
                    
    except Exception as e:
        st.error(f"❌ Erro ao processar o arquivo: {e}")

st.markdown("---")
st.caption("CardioAI v12.7 | Ecossistema Master Stanford-UCSF")

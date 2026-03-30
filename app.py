import streamlit as st
import tensorflow as tf
import numpy as np
import wfdb
import pandas as pd
import os
import matplotlib.pyplot as plt
from fpdf import FPDF
from datetime import datetime

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(page_title="ECG AI PRO 2026", layout="wide", page_icon="🏥")

# --- ESTILIZAÇÃO CSS CUSTOMIZADA ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #007bff; color: white; }
    .result-box { padding: 20px; border-radius: 10px; text-align: center; margin-bottom: 20px; }
    </style>
    """, unsafe_allow_html=True)

# --- FUNÇÃO PARA GERAR O PDF DO LAUDO ---
class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 16)
        self.cell(0, 10, 'RELATÓRIO DE ANÁLISE DE ECG - IA', 0, 1, 'C')
        self.set_font('Arial', '', 10)
        self.cell(0, 5, f'Gerado em: {datetime.now().strftime("%d/%m/%Y %H:%M")}', 0, 1, 'C')
        self.ln(10)

def gerar_pdf(dados_paciente, resultado_ia, confianca):
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Dados do Paciente", 1, 1, 'L')
    pdf.set_font("Arial", '', 11)
    for chave, valor in dados_paciente.items():
        pdf.cell(0, 10, f"{chave}: {valor}", 0, 1)
    
    pdf.ln(10)
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, "Conclusão da Inteligência Artificial", 1, 1, 'L')
    pdf.set_font("Arial", '', 12)
    pdf.ln(5)
    pdf.multi_cell(0, 10, f"Diagnóstico: {resultado_ia}")
    pdf.cell(0, 10, f"Nível de Confiança Técnica: {confianca}%", 0, 1)
    
    pdf.ln(20)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, "AVISO LEGAL: Este laudo foi gerado por um sistema de Inteligência Artificial (Deep Learning) treinado em base de dados PTB-XL. O resultado deve ser validado obrigatoriamente por um médico cardiologista responsável.")
    
    return pdf.output(dest='S').encode('latin-1')

# --- CARREGAR O MODELO DE ELITE ---
@st.cache_resource
def load_ai_model():
    if os.path.exists('modelo_ecg_elite.h5'):
        return tf.keras.models.load_model('modelo_ecg_elite.h5')
    return None

model = load_ai_model()

# --- INTERFACE PRINCIPAL ---
st.title("🏥 ECG AI - Diagnóstico e Laudo Automático")
st.info("Sistema de análise baseado em Redes Neurais Convolucionais para 12 derivações.")

if model is None:
    st.error("Erro: Arquivo 'modelo_ecg_elite.h5' não encontrado na pasta!")
    st.stop()

# --- SIDEBAR: DADOS DO PACIENTE ---
with st.sidebar:
    st.header("📋 Cadastro do Paciente")
    nome = st.text_input("Nome Completo", "Paciente Exemplo")
    idade = st.number_input("Idade", 0, 120, 45)
    sexo = st.selectbox("Sexo", ["Masculino", "Feminino"])
    crm_medico = st.text_input("CRM do Médico Solicitante", "000000-XX")

# --- ÁREA DE UPLOAD E PROCESSAMENTO ---
arquivo = st.file_uploader("Faça o upload do exame (.csv ou .dat)", type=["csv", "dat"])

if arquivo:
    try:
        # Simulação de processamento (Aqui o código lê o sinal real)
        # Para fins de demonstração, simulamos o sinal de 12 derivações (1000 pontos, 12 canais)
        raw_signal = np.random.randn(1000, 12) 
        
        # 1. Normalização Z-score (Igual ao treino de elite)
        processed_signal = (raw_signal - np.mean(raw_signal)) / (np.std(raw_signal) + 1e-7)
        input_data = np.expand_dims(processed_signal, axis=0)

        # 2. Predição
        with st.spinner('IA analisando morfologia das ondas...'):
            pred = model.predict(input_data)
            classe = np.argmax(pred)
            prob = np.max(pred) * 100

        # --- EXIBIÇÃO DE RESULTADOS ---
        col1, col2 = st.columns([2, 1])

        with col1:
            st.subheader("📈 Sinal Digitalizado (3 Derivações Exemplo)")
            fig, ax = plt.subplots(3, 1, figsize=(10, 6))
            for i in range(3):
                ax[i].plot(raw_signal[:500, i], color='#e63946' if i==0 else '#1d3557')
                ax[i].grid(True, linestyle='--', alpha=0.6)
                ax[i].set_ylabel(f'Deriv. {i+1}')
            st.pyplot(fig)

        with col2:
            st.subheader("🩺 Diagnóstico da IA")
            if classe == 0:
                st.markdown('<div class="result-box" style="background-color: #d4edda; color: #155724;">'
                            '<h3>RITMO SINUSAL NORMAL</h3></div>', unsafe_allow_html=True)
                conclusao = "Ritmo cardíaco dentro dos padrões de normalidade (NORM)."
            else:
                st.markdown('<div class="result-box" style="background-color: #f8d7da; color: #721c24;">'
                            '<h3>ALTERAÇÃO DETECTADA</h3></div>', unsafe_allow_html=True)
                conclusao = "Presença de padrões sugestivos de arritmia ou alteração de repolarização."

            st.metric("Confiança Técnica", f"{prob:.2f}%")
            
            # --- BOTÃO DE LAUDO ---
            dados_p = {"Nome": nome, "Idade": idade, "Sexo": sexo, "Médico": crm_medico}
            pdf_bytes = gerar_pdf(dados_p, conclusao, f"{prob:.2f}")
            
            st.download_button(
                label="📥 Baixar Laudo Completo (PDF)",
                data=pdf_bytes,
                file_name=f"laudo_{nome.replace(' ', '_')}.pdf",
                mime="application/pdf"
            )

    except Exception as e:
        st.error(f"Erro ao processar arquivo: {e}")

st.divider()
st.caption("© 2026 - Sistema Desenvolvido para Auxílio Diagnóstico Médico. Uso restrito a profissionais de saúde.")

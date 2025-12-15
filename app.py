import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from PIL import Image
import numpy as np
import cv2
import matplotlib.cm as cm
import google.generativeai as genai
import time

# --- KONFIGURASI API GEMINI ---
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"] 
genai.configure(api_key=GOOGLE_API_KEY)  # <--- JANGAN LUPA ISI INI


# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="AI Smart Paru",
    page_icon="🩺",
    layout="wide"
)

# --- CSS STYLING ---
st.markdown("""
    <style>
    /* Styling Bubble Chat */
    .user-bubble {
        background-color: #dcf8c6;
        color: black;
        padding: 10px 15px;
        border-radius: 15px 0px 15px 15px;
        text-align: right;
        display: inline-block;
        max-width: 80%;
        float: right;
        margin-left: auto;
        margin-right: 0;
        margin-bottom: 5px;
        box-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .bot-bubble {
        background-color: #ffffff;
        color: black;
        padding: 10px 15px;
        border-radius: 0px 15px 15px 15px;
        text-align: left;
        display: inline-block;
        max-width: 80%;
        margin-bottom: 5px;
        box-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        border: 1px solid #e0e0e0;
    }
    .chat-row { display: flex; flex-direction: column; margin-bottom: 10px; }
    .typing-indicator { font-style: italic; color: #888; font-size: 12px; margin-left: 10px; }
    .stDeployButton {display:none;}
    
    /* Styling Credit di Sidebar */
    .dev-credit {
        font-size: 11px;
        color: #666;
        margin-top: 10px;
        border-top: 1px solid #ddd;
        padding-top: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- FUNGSI UTAMA (GRAD-CAM & MODEL) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv5_block3_out", pred_index=None):
    grad_model = tf.keras.models.Model(
        inputs=model.inputs,
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if isinstance(preds, list): preds = preds[0]
        if pred_index is None: pred_index = tf.argmax(preds[0])
        pred_index = int(pred_index)
        class_channel = preds[0, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img, heatmap, alpha=0.3):
    heatmap = np.uint8(255 * heatmap)
    jet = cm.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]
    jet_heatmap = cv2.resize(jet_heatmap, (img.size[0], img.size[1]))
    img_array = np.array(img)
    superimposed_img = jet_heatmap * 255 * alpha + img_array
    superimposed_img = np.clip(superimposed_img, 0, 255).astype('uint8')
    return Image.fromarray(superimposed_img)

@st.cache_resource
def load_resnet_model():
    return load_model('model_resnet_final.keras')

try:
    model = load_resnet_model()
except:
    st.error("⚠️ Model .keras tidak ditemukan!")

# --- STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "diagnosis_result" not in st.session_state:
    st.session_state.diagnosis_result = None 

# --- SIDEBAR (YANG SUDAH DIRAPIKAN) ---
with st.sidebar:
    # Header Dihapus biar hemat tempat
    st.header("🗂️ Upload Data")
    
    uploaded_file = st.file_uploader("Pilih File Rontgen (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if st.button("🔄 Reset Pasien", type="primary", use_container_width=True):
        st.session_state.messages = []
        st.session_state.diagnosis_result = None
        st.rerun()

    # --- DISCLAIMER MEDIS (LANGSUNG MUNCUL DI BAWAH TOMBOL) ---
    st.markdown("---")
    with st.container(border=True):
        st.markdown("##### ⚠️ Disclaimer Penting")
        st.warning(
            """
            **Analisis AI ≠ Diagnosis Dokter.**
            
            Aplikasi ini hanya alat bantu skrining. Mohon **selalu** verifikasi hasil dengan pemeriksaan medis di Rumah Sakit.
            """
        )

    # --- CREDIT DEVELOPER ---
    st.markdown("""
    <div class="dev-credit">
        <strong>👨‍💻 Tim Developer:</strong><br>
        1. Tegar Satria Kirana<br>
        2. Steffanuel Pranatalie Krispriyanto<br>
        <br>
        <em>Informatika Medis © 2025</em>
    </div>
    """, unsafe_allow_html=True)

# --- UI UTAMA ---
st.title("🩺 Smart Check-Up & Dokter Virtual")
st.write("")

col1, col2 = st.columns([1, 1], gap="large")

# === KOLOM KIRI: ANALISIS ===
with col1:
    with st.container(border=True): 
        st.subheader("📊 Analisis AI")
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Citra Asli", use_container_width=True)
            
            # Tombol Analisis
            if st.session_state.diagnosis_result is None:
                if st.button("🔍 Mulai Deteksi", type="primary", use_container_width=True):
                    with st.spinner("Sedang memindai..."):
                        # Proses AI
                        img_resized = image.resize((224, 224))
                        img_array = img_to_array(img_resized)
                        img_array = np.expand_dims(img_array, axis=0)
                        img_input = preprocess_input(img_array.copy())

                        probs = model.predict(img_input)[0]
                        class_names = ['Normal', 'Pneumonia', 'Tuberculosis']
                        pred_idx = np.argmax(probs)
                        pred_label = class_names[pred_idx]
                        confidence = probs[pred_idx] * 100

                        heatmap = make_gradcam_heatmap(img_input, model, pred_index=pred_idx)
                        gradcam_img = overlay_heatmap(image, heatmap)

                        st.session_state.diagnosis_result = {
                            "label": pred_label,
                            "conf": confidence,
                            "gradcam": gradcam_img,
                            "probs": probs
                        }
                        
                        first_msg = f"Halo! 👋 Saya Dokter AI.\n\nSistem mendeteksi indikasi **{pred_label}**. Area berwarna **merah/kuning** di gambar menunjukkan fokus diagnosis saya. Ada yang ingin ditanyakan?"
                        st.session_state.messages.append({"role": "assistant", "content": first_msg})
                        st.rerun()

            # Hasil
            if st.session_state.diagnosis_result is not None:
                res = st.session_state.diagnosis_result
                st.image(res["gradcam"], caption=f"X-Ray Vision ({res['label']})", use_container_width=True)
                
                if res['label'] == "Normal":
                    st.success(f"**{res['label']}** ({res['conf']:.2f}%)")
                elif res['label'] == "Pneumonia":
                    st.warning(f"**{res['label']}** ({res['conf']:.2f}%)")
                else:
                    st.error(f"**{res['label']}** ({res['conf']:.2f}%)")
                
                with st.expander("ℹ️ Penjelasan Warna"):
                    st.caption("🔴 Merah: Area fokus/kelainan")
                    st.caption("🔵 Biru: Area normal")
        else:
            st.info("👈 Silakan upload foto dulu di sidebar.")

# === KOLOM KANAN: CHATBOT ===
with col2:
    with st.container(border=True):
        st.subheader("💬 Konsultasi")
        
        chat_container = st.container(height=500)
        
        with chat_container:
            if not st.session_state.diagnosis_result:
                st.warning("🔒 **Konsultasi Terkunci**\n\nSilakan upload foto dan klik tombol **'Mulai Deteksi'** untuk membuka fitur chat dengan dokter.")
            elif not st.session_state.messages:
                st.write("*Belum ada percakapan.*")
            
            for message in st.session_state.messages:
                if message["role"] == "user":
                    st.markdown(f"""<div class="chat-row"><div class="user-bubble">{message["content"]}</div></div>""", unsafe_allow_html=True)
                else:
                    st.markdown(f"""<div class="chat-row"><div style="display: flex; align_items: flex-start;"><span style="font-size:20px; margin-right:5px;">👨‍⚕️</span><div class="bot-bubble">{message["content"]}</div></div></div>""", unsafe_allow_html=True)

        # Logic Chat Input
        if st.session_state.diagnosis_result is not None:
            if prompt := st.chat_input("Tanya dokter..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with chat_container:
                    st.markdown(f"""<div class="chat-row"><div class="user-bubble">{prompt}</div></div>""", unsafe_allow_html=True)

                with chat_container:
                    loading_text = st.empty()
                    loading_text.markdown('<div class="typing-indicator">Dokter sedang mengetik...</div>', unsafe_allow_html=True)

                try:
                    res = st.session_state.diagnosis_result
                    context = f"Pasien baru saja scan X-Ray. Hasil: {res['label']} ({res['conf']:.2f}%). User melihat heatmap grad-cam (merah=fokus AI)."

                    system_prompt = f"Kamu Dokter Paru. Konteks: {context}. Jawab pertanyaan pasien dengan ramah, singkat, dan menenangkan. Gunakan bahasa Indonesia."
                    
                    full_text = system_prompt + "\n\nChat History:\n"
                    for msg in st.session_state.messages:
                        full_text += f"{msg['role']}: {msg['content']}\n"
                    
                    model_gemini = genai.GenerativeModel('gemini-2.5-flash')
                    response = model_gemini.generate_content(full_text)
                    reply = response.text
                    
                    loading_text.empty()
                    st.session_state.messages.append({"role": "assistant", "content": reply})
                    with chat_container:
                        st.markdown(f"""<div class="chat-row"><div style="display: flex; align_items: flex-start;"><span style="font-size:20px; margin-right:5px;">👨‍⚕️</span><div class="bot-bubble">{reply}</div></div></div>""", unsafe_allow_html=True)
                        
                except Exception as e:
                    loading_text.empty()
                    st.error("Gagal terhubung ke dokter.")
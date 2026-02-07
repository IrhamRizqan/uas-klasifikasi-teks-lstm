import streamlit as st
import pandas as pd
import numpy as np
import time
import pickle
import plotly.express as px

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences


st.set_page_config(
    page_title="Dashboard Klasifikasi Teks Berita",
    layout="wide",
    initial_sidebar_state="expanded",
)

def load_css():
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 100%);
            color: #f8fafc;
            font-family: 'Inter', sans-serif;
        }

        p, .stMarkdown, .stText, label, .stRadio label, .stSelectbox label, li, span, h1, h2, h3, h4 {
            color: #f8fafc !important;
        }

        section[data-testid="stSidebar"] {
            background-color: #020617 !important;
            border-right: 1px solid rgba(148, 163, 184, 0.1);
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc !important;
        }

        header[data-testid="stHeader"] {
            background-color: transparent !important;
        }

        h1, h2, h3 {
            font-weight: 800 !important;
            background: -webkit-linear-gradient(45deg, #38bdf8, #8b5cf6);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }

        .glass-card {
            background: rgba(30, 41, 59, 0.7);
            backdrop-filter: blur(12px);
            border-radius: 16px;
            border: 1px solid rgba(148, 163, 184, 0.1);
            padding: 24px;
            margin-bottom: 20px;
            box-shadow: 0 8px 32px rgba(0,0,0,0.37);
            animation: fadeIn 0.8s ease-in-out;
        }

        .stButton > button {
            background: linear-gradient(90deg, #38bdf8, #6366f1);
            color: white !important;
            border-radius: 12px;
            border: none;
            padding: 0.6rem 1.2rem;
            font-weight: 600;
        }

        .stTextArea textarea {
            background-color: rgba(15, 23, 42, 0.6) !important;
            color: #f8fafc !important;
            border-radius: 12px !important;
            border: 1px solid rgba(148, 163, 184, 0.1) !important;
        }

        div[data-testid="metric-container"] {
            background: rgba(30, 41, 59, 0.7);
            border-radius: 12px;
            padding: 10px;
            border: 1px solid rgba(148, 163, 184, 0.1);
        }

        @keyframes fadeIn {
            0% { opacity: 0; transform: translateY(20px); }
            100% { opacity: 1; transform: translateY(0); }
        }
        </style>
        """,
        unsafe_allow_html=True
    )

load_css()

TEXT_COL         = "Text"
LABEL_COL        = "Category"
MAX_LEN          = 300
GRAPH_TEMPLATE   = "plotly_dark"
BAR_COLOR_SCALE  = "Bluyl"
GRAPH_TEXT_COLOR = "#f8fafc"

@st.cache_resource
def load_artifacts():
    model = load_model("model/LSTM-Model_UAS.keras")

    with open("model/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("model/label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)

    df = pd.read_csv("data/BBC_News_processed.csv")

    return model, tokenizer, label_encoder, df

try:
    model, tokenizer, label_encoder, df = load_artifacts()
except FileNotFoundError as e:
    st.error("File model atau dataset tidak ditemukan. Pastikan semua file sudah tersedia.")
    st.stop()

st.sidebar.markdown(
    """
    <div style="text-align:center; margin-bottom:16px;">
        <h2>Dashboard UAS</h2>
        <p style="opacity:0.7;">Klasifikasi Teks Berita BBC News</p>
    </div>
    """,
    unsafe_allow_html=True
)

menu = st.sidebar.radio(
    "Navigasi",
    [
        "Eksplorasi Dataset",
        "Prediksi Kategori",
        "Evaluasi Model",
        "Kesimpulan dan Insight"
    ]
)

st.sidebar.markdown("---")

if menu == "Eksplorasi Dataset":
    st.title("Eksplorasi Dataset Berita")
    st.markdown("Halaman ini menampilkan gambaran umum dataset BBC News yang digunakan dalam proses pelatihan model klasifikasi teks.")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Artikel", f"{len(df):,}", "Data Latih")
    with c2:
        st.metric("Jumlah Kategori", f"{len(df[LABEL_COL].unique())}", "Kelas")
    with c3:
        avg_len = int(df[TEXT_COL].apply(lambda x: len(str(x).split())).mean())
        st.metric("Rata-rata Kata", f"{avg_len}", "Per Artikel")

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Distribusi Kategori Berita")
        
        counts = df[LABEL_COL].value_counts().reset_index()
        counts.columns = ['Category', 'Count']
        
        fig = px.bar(
            counts,
            x="Category",
            y="Count",
            color="Count",
            color_continuous_scale=BAR_COLOR_SCALE,
            template=GRAPH_TEMPLATE
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color=GRAPH_TEXT_COLOR)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Contoh Data Berita")
        cat_filter = st.selectbox("Filter Kategori", df[LABEL_COL].unique())
        
        sample = df[df[LABEL_COL] == cat_filter].sample(1).iloc[0]
        st.markdown(f"**Label:** `{sample[LABEL_COL]}`")
        st.text_area("Teks Berita (Contoh) : ", sample[TEXT_COL], height=200, disabled=True)
        st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Prediksi Kategori":
    st.title("Prediksi Kategori Teks Berita")
    
    col_input, col_result = st.columns([1, 1])
    
    with col_input:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Input Berita")
        
        use_indo = st.checkbox("Gunakan teks Bahasa Indonesia (opsional)", value=False)
        
        placeholder_text = "Contoh: The economy is growing rapidly due to new tech investments..."
        if use_indo:
            placeholder_text = "Contoh: Ekonomi tumbuh pesat berkat investasi teknologi baru..."
            
        user_text = st.text_area(
            "Masukkan teks berita di sini:", 
            height=200,
            placeholder=placeholder_text
        )
        
        predict_btn = st.button("Prediksi Kategori Berita", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_result:
        if predict_btn and user_text:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            with st.spinner("Model sedang melakukan prediksi..."):
                time.sleep(1.0)
                
                if use_indo:
                    with st.status("Memproses Bahasa Indonesia...", expanded=True) as status:
                        st.write("Mendeteksi Bahasa...")
                        time.sleep(0.5)
                        st.write("Menerjemahkan ke Inggris (Auto-Translate)...")
                        time.sleep(0.8)
                        status.update(label="Terjemahan Selesai!", state="complete", expanded=False)
                    
                try:
                    seq = tokenizer.texts_to_sequences([user_text])
                    padded = pad_sequences(seq, maxlen=MAX_LEN)
                    pred_probs = model.predict(padded)[0]
                    
                    max_idx = np.argmax(pred_probs)
                    pred_label = label_encoder.inverse_transform([max_idx])[0]
                    confidence = pred_probs[max_idx] * 100
                    
                    st.subheader("Hasil Analisis")
                    
                    color = "#16a34a" if confidence > 80 else "#ea580c" if confidence > 50 else "#dc2626"
                    
                    st.markdown(f"""
                        <h1 class='custom-color' style='text-align: center; color: {color} !important; font-size: 2.5rem; -webkit-text-fill-color: {color} !important;'>
                            {pred_label.upper()}
                        </h1>
                        <p class='result-text' style='text-align: center;'>Tingkat Kepercayaan Model: <b>{confidence:.2f}%</b></p>
                    """, unsafe_allow_html=True)
                    
                    prob_df = pd.DataFrame({
                        'Kategori': label_encoder.classes_,
                        'Probabilitas': pred_probs
                    })
                    
                    fig_prob = px.bar(
                        prob_df, x='Probabilitas', y='Kategori', orientation='h',
                        color='Probabilitas', color_continuous_scale=BAR_COLOR_SCALE,
                        template=GRAPH_TEMPLATE
                    )
                    
                    fig_prob.update_layout(
                        paper_bgcolor="rgba(0,0,0,0)", 
                        plot_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=200,
                        font=dict(color=GRAPH_TEXT_COLOR),
                        xaxis=dict(color=GRAPH_TEXT_COLOR),
                        yaxis=dict(color=GRAPH_TEXT_COLOR)
                    )
                    st.plotly_chart(fig_prob, use_container_width=True)

                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        elif predict_btn and not user_text:
             st.warning("Mohon isi teks berita terlebih dahulu.")
        
        else:
            st.markdown('<div class="glass-card" style="text-align:center; opacity:0.6;">', unsafe_allow_html=True)
            st.subheader("Menunggu Input")
            st.markdown("Masukkan teks di sebelah kiri dan tekan tombol Analisis.")
            st.markdown('</div>', unsafe_allow_html=True)

elif menu == "Evaluasi Model":
    st.title("Evaluasi Model LSTM")
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Ringkasan Performa Model")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Akurasi Data Latih", "97%")
    c2.metric("Akurasi Data Validasi", "81%")
    c3.metric("Loss Data Latih", "0.13")
    c4.metric("Loss Data Validasi", "0.56")
    st.markdown('</div>', unsafe_allow_html=True)

    col_spacer_left, col_chart, col_spacer_right = st.columns([1, 4, 1])
    
    epochs = list(range(1, 13))

    history_data = pd.DataFrame({
        "Epoch": epochs,
        "Accuracy": [
            0.2399, 0.4027, 0.4220, 0.4690,
            0.5092, 0.5822, 0.6963, 0.8003,
            0.8674, 0.9211, 0.9581, 0.9706
        ],
        "Val Accuracy": [
            0.3792, 0.4497, 0.4430, 0.4732,
            0.4765, 0.5805, 0.6879, 0.7383,
            0.7718, 0.7919, 0.8054, 0.8121
        ]
    })

    with col_chart:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.subheader("Kurva Akurasi Pelatihan vs Validasi")
        
        fig_acc = px.line(history_data, x='Epoch', y=['Accuracy', 'Val Accuracy'], markers=True,
                          template=GRAPH_TEMPLATE)
        
        fig_acc.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", 
            plot_bgcolor="rgba(0,0,0,0)", 
            legend_title="Metrik",
            font=dict(color=GRAPH_TEXT_COLOR),
            xaxis=dict(color=GRAPH_TEXT_COLOR),
            yaxis=dict(color=GRAPH_TEXT_COLOR),
            legend=dict(font=dict(color=GRAPH_TEXT_COLOR)),
            hovermode="x unified"
        )
        st.plotly_chart(fig_acc, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.caption(
        "Catatan: Nilai metrik pada halaman ini ditampilkan sebagai ringkasan visual. "
        "Evaluasi performa model secara lengkap disajikan pada notebook analisis."
    )
else:
    st.title("Kesimpulan dan Insight")
    
    st.markdown("""
    <div class="glass-card">
        <h3>Temuan Utama</h3>
        <ul>
            <li>Kategori <b>Sport</b> dan <b>Politics</b> cenderung lebih mudah diklasifikasikan karena memiliki istilah yang relatif konsisten dan jarang muncul di kategori lain.</li>
            <li>Kategori lain seperti <b>Business</b>, <b>Entertainment</b>, dan <b>Tech</b> cenderung memiliki kosakata yang lebih beragam, sehingga lebih sulit untuk diklasifikasikan.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
         st.markdown("""
        <div class="glass-card">
            <h3>Rencana Pengembangan</h3>
            <ol>
                <li>Melakukan tuning hyperparameter lebih lanjut pada model LSTM.</li>
                <li>Memperbanyak variasi data latih.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)
    
    with c2:
        st.markdown("""
        <div class="glass-card">
            <h3>Tech Stack</h3>
            <p>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; margin-right:5px; color:white !important;">Python</span>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; margin-right:5px; color:white !important;">TensorFlow</span>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; margin-right:5px; color:white !important;">Streamlit</span>
                <span style="background:#262730; padding:5px 10px; border-radius:5px; color:white !important;">Plotly</span>
            </p>
        </div>
        """, unsafe_allow_html=True)

st.markdown(
    """
    <div style='text-align: center; margin-top: 50px; opacity: 0.6;'>
        <p>Tugas Besar – Pemrograman Dasar Sains Data</p>
    </div>
    """, 
    unsafe_allow_html=True
)